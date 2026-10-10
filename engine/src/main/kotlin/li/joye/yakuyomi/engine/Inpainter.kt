package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import kotlin.math.abs
import kotlin.math.roundToInt
import kotlinx.coroutines.coroutineScope

/**
 * 去字（text removal）。真機 A/B 定案＝兩門別，皆純 NCNN（`.param`/`.bin`）：
 *  - **boxfill（快速去字）**：逐區把去字遮罩像素平塗成背景色。瞬間、不跑模型；乾淨白泡完美、壓畫面是平色塊。
 *  - **aot（AI 去字·預設）**：AOT-GAN（m-i-t 漫畫權重）整頁一次縮到 [InpainterConfig.tileSize]（768）重建背景、全區重建。
 *    全卷積 → 任意尺寸；768＝畫質/記憶體/藏在翻譯下（§8 去字‖翻譯重疊）的甜蜜點。
 *
 * AOT I/O 契約（對齊 parity/inpaint_parity.py）：img∈[-1,1] 且洞歸零（m-i-t `img*(1-mask)`）、mask∈{0,1}(1=擦)、輸出∈[-1,1]。
 * LaMa（整頁縮 512 必糊）與 AOT 逐格（原生解析度·CPU 太貴）皆已退役移除；GPU/Vulkan 實測算不對 AOT-GAN（見 memory ncnn-vulkan-fp16）。
 */
class Inpainter(
    modelPath: String,
    private val cfg: InpainterConfig = InpainterConfig(),
) : AutoCloseable {

    private var ncnnHandle: Long = 0L
    /** 實際生效的後端；無 adb 時由呼叫端寫進 log/圖確認。 */
    val ep: String = "NCNN-CPU"

    init {
        check(modelPath.endsWith(".param")) { "去字需 NCNN `.param` 模型（AOT-GAN）：$modelPath" }
        check(NcnnBackend.available) { "NCNN 原生庫未載入，無法去字" }
        val bin = modelPath.removeSuffix(".param") + ".bin"
        ncnnHandle = NcnnBackend.createNet(modelPath, bin)
        check(ncnnHandle != 0L) { "NCNN AOT 模型載入失敗：$modelPath" }
    }

    suspend fun inpaint(page: Bitmap, regions: List<TextRegion>, textMask: Bitmap): Bitmap = coroutineScope {
        val w = page.width
        val h = page.height
        val result = page.copy(Bitmap.Config.ARGB_8888, true)
        // seg 細筆畫遮罩 ∩ 保留區 bbox（外擴 bboxPad）再膨脹：只動筆畫、限制在翻譯過的區（SFX/未譯留原圖，§11）。
        val maskPx = buildSegMask(regions, textMask, w, h)

        if (cfg.method == "boxfill") {
            // 逐區平塗背景色：白泡乾淨無殘留、忙碌區是平色塊（要品質用 aot）。
            val px = IntArray(w * h); result.getPixels(px, 0, w, 0, 0, w, h)
            val tightPx = IntArray(w * h); textMask.getPixels(tightPx, 0, w, 0, 0, w, h)
            for (r in regions) {
                val s = bgStats(px, tightPx, r, w, h)
                r.onArt = false; r.dbgStd = s.std; r.dbgWhite = s.meanLum // dbg 值給 sandbox 去背比較標框
                flatFill(result, maskPx, r, s.color, cfg.bboxPad, w, h)
            }
            return@coroutineScope result
        }

        // aot（預設）：全區都跑 AOT-GAN 整頁重建；標 onArt 讓 Renderer 給黑字粗白邊。
        regions.forEach { it.onArt = true }
        val maskBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        maskBmp.setPixels(maskPx, 0, w, 0, 0, w, h)
        if (maxOf(w, h) > TALL_RATIO * minOf(w, h)) {
            // 長條（條漫／webtoon）：整頁壓成方形 tileSize 會把文字壓扁好幾倍 → AOT 重建出橫紋色帶。
            // 改成：平底區直接平塗背景色；其餘區各自取「短邊見方」的視窗跑 AOT（比例不失真）。
            inpaintTall(page, result, regions, maskPx, maskBmp, textMask, w, h)
        } else {
            runWholeAot(page, maskBmp, w, h)?.let { compositePixels(result, maskPx, it) }
        }
        maskBmp.recycle()
        result
    }

    /**
     * 長條頁去字：
     *  1. 區內背景近乎單色（眾數色佔背景 ≥ [FLAT_FRAC]，乾淨白泡／色框）→ 平塗眾數色（同 boxfill，瞬間且無殘影），onArt=false。
     *  2. 其餘（壓畫面）→ 以區中心取「短邊×短邊」視窗（區比短邊大才放大視窗）跑 AOT，只把該區（bbox 外擴）內的遮罩像素貼回。
     *     視窗彼此涵蓋的區合併成一次推論。
     */
    private fun inpaintTall(
        page: Bitmap, result: Bitmap, regions: List<TextRegion>, maskPx: IntArray, maskBmp: Bitmap,
        textMask: Bitmap, w: Int, h: Int,
    ) {
        val px = IntArray(w * h); page.getPixels(px, 0, w, 0, 0, w, h)
        val tightPx = IntArray(w * h); textMask.getPixels(tightPx, 0, w, 0, 0, w, h)
        val pad = cfg.bboxPad + (cfg.maskDilate / 2f).roundToInt() + 1
        val busy = ArrayList<TextRegion>()
        for (r in regions) {
            val s = bgStats(px, tightPx, r, w, h)
            r.dbgStd = s.std; r.dbgWhite = s.meanLum
            if (s.modeFrac >= FLAT_FRAC) {
                r.onArt = false
                flatFill(result, maskPx, r, s.modeColor, pad, w, h)
            } else {
                busy += r
            }
        }
        if (busy.isEmpty()) return

        val side = minOf(w, h)
        class Win(val x0: Int, val y0: Int, val ww: Int, val wh: Int, val members: MutableList<TextRegion> = ArrayList())
        val wins = ArrayList<Win>()
        for (r in busy.sortedBy { it.y0 }) {
            val rx0 = (r.x0.toInt() - pad).coerceAtLeast(0); val ry0 = (r.y0.toInt() - pad).coerceAtLeast(0)
            val rx1 = (r.x1.toInt() + pad).coerceAtMost(w); val ry1 = (r.y1.toInt() + pad).coerceAtMost(h)
            val host = wins.firstOrNull { rx0 >= it.x0 && ry0 >= it.y0 && rx1 <= it.x0 + it.ww && ry1 <= it.y0 + it.wh }
            if (host != null) { host.members += r; continue }
            val ww = maxOf(side, rx1 - rx0 + 2 * pad).coerceAtMost(w)
            val wh = maxOf(side, ry1 - ry0 + 2 * pad).coerceAtMost(h)
            val cx = (rx0 + rx1) / 2; val cy = (ry0 + ry1) / 2
            val x0 = (cx - ww / 2).coerceIn(0, w - ww); val y0 = (cy - wh / 2).coerceIn(0, h - wh)
            wins += Win(x0, y0, ww, wh).also { it.members += r }
        }
        for (win in wins) {
            val cropPage = Bitmap.createBitmap(page, win.x0, win.y0, win.ww, win.wh)
            val cropMask = Bitmap.createBitmap(maskBmp, win.x0, win.y0, win.ww, win.wh)
            try {
                val o = runWholeAot(cropPage, cropMask, win.ww, win.wh) ?: continue
                val cur = IntArray(win.ww * win.wh)
                result.getPixels(cur, 0, win.ww, win.x0, win.y0, win.ww, win.wh)
                for (r in win.members) {
                    val ax0 = ((r.x0.toInt() - pad) - win.x0).coerceIn(0, win.ww)
                    val ay0 = ((r.y0.toInt() - pad) - win.y0).coerceIn(0, win.wh)
                    val ax1 = ((r.x1.toInt() + pad) - win.x0).coerceIn(0, win.ww)
                    val ay1 = ((r.y1.toInt() + pad) - win.y0).coerceIn(0, win.wh)
                    for (y in ay0 until ay1) {
                        val mrow = (win.y0 + y) * w + win.x0
                        val row = y * win.ww
                        for (x in ax0 until ax1) if ((maskPx[mrow + x] and 0xFF) > 127) cur[row + x] = o.px[row + x]
                    }
                }
                result.setPixels(cur, 0, win.ww, win.x0, win.y0, win.ww, win.wh)
            } finally {
                if (cropPage !== page) cropPage.recycle()
                if (cropMask !== maskBmp) cropMask.recycle()
            }
        }
    }

    /**
     * 暖機：aot 方法對空白小圖跑一次 NCNN AOT session（首次 lazy 初始化在單緒完成）。
     * boxfill 只平塗、不跑該 session → 無需暖（也不會冷撞）。併發翻多頁前先呼叫一次。
     */
    fun warmUp() {
        if (cfg.method == "boxfill") return
        val w = 64
        val h = 64
        val page = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val mask = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        try {
            runWholeAot(page, mask, w, h)
        } catch (t: Throwable) {
            Log.w(TAG, "去字暖機失敗：${t.message}")
        } finally {
            page.recycle()
            mask.recycle()
        }
    }

    /** 去字遮罩 Bitmap（白＝要去字）。給重繪素材/視覺化用；與 inpaint 同一份 seg 細遮罩。 */
    fun buildMask(page: Bitmap, regions: List<TextRegion>, textMask: Bitmap): Bitmap {
        val w = page.width; val h = page.height
        val maskPx = buildSegMask(regions, textMask, w, h)
        return Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).apply { setPixels(maskPx, 0, w, 0, 0, w, h) }
    }

    /**
     * seg 細遮罩＝seg 細筆畫 ∩ 已保留區的「區域 bbox 矩形（外擴 bboxPad）」（allow），再膨脹。回傳 ARGB 像素（白＝要去字）。
     * ★ 用 bbox 矩形(非緊的文字行框)＝涵蓋漢字旁的注音假名；pad 再外擴涵蓋貼 bbox 邊界的假名（桌面 auto_diag.py 實證）。
     */
    private fun buildSegMask(regions: List<TextRegion>, textMask: Bitmap, w: Int, h: Int): IntArray {
        val pad = cfg.bboxPad
        val allow = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        Canvas(allow).apply {
            drawColor(Color.BLACK)
            val p = Paint().apply { color = Color.WHITE; style = Paint.Style.FILL }
            for (region in regions) drawRect(region.x0 - pad, region.y0 - pad, region.x1 + pad, region.y1 + pad, p)
        }
        val mask = IntArray(w * h)
        allow.getPixels(mask, 0, w, 0, 0, w, h)
        allow.recycle()
        val seg = IntArray(w * h)
        textMask.getPixels(seg, 0, w, 0, 0, w, h)
        for (i in mask.indices) {
            mask[i] = if ((mask[i] and 0xFF) > 127 && (seg[i] and 0xFF) > 127) Color.WHITE else Color.BLACK
        }
        dilate(mask, w, h, (cfg.maskDilate / 2f).roundToInt().coerceAtLeast(1))
        return mask
    }

    /** 二值遮罩可分離膨脹（先橫後縱 max-filter），radius 像素。覆蓋筆畫抗鋸齒邊緣、給去字餘裕。 */
    private fun dilate(px: IntArray, w: Int, h: Int, radius: Int) {
        if (radius <= 0) return
        val tmp = IntArray(px.size)
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                var on = false; var k = -radius
                while (k <= radius) { val xx = x + k; if (xx in 0 until w && (px[row + xx] and 0xFF) > 127) { on = true; break }; k++ }
                tmp[row + x] = if (on) Color.WHITE else Color.BLACK
            }
        }
        for (x in 0 until w) {
            for (y in 0 until h) {
                var on = false; var k = -radius
                while (k <= radius) { val yy = y + k; if (yy in 0 until h && (tmp[yy * w + x] and 0xFF) > 127) { on = true; break }; k++ }
                px[y * w + x] = if (on) Color.WHITE else Color.BLACK
            }
        }
    }

    /** [modeColor]／[modeFrac]：背景最常見的顏色（5-bit 量化分箱的平均）與落在該色 ±[MODE_TOL] 內的背景像素比例。 */
    private class BgStat(val meanLum: Float, val std: Float, val color: Int, val modeColor: Int = color, val modeFrac: Float = 1f)

    /**
     * 區 bbox 內「非文字(背景)」像素的亮度均值+std+平均色。tightPx＝未膨脹 textMask（量得到筆畫間的白）。
     * boxfill 用 [BgStat.color] 平塗；std/meanLum 只給 sandbox 去背比較標框（對照用）。用行框多邊形局部遮罩避開軸對齊 bbox 角落雜訊。
     */
    private fun bgStats(px: IntArray, tightPx: IntArray, region: TextRegion, w: Int, h: Int): BgStat {
        val x0 = region.x0.toInt().coerceIn(0, w - 1)
        val y0 = region.y0.toInt().coerceIn(0, h - 1)
        val x1 = region.x1.toInt().coerceIn(x0 + 1, w)
        val y1 = region.y1.toInt().coerceIn(y0 + 1, h)
        val bw = x1 - x0; val bh = y1 - y0
        val qmBmp = Bitmap.createBitmap(bw, bh, Bitmap.Config.ARGB_8888)
        Canvas(qmBmp).apply {
            drawColor(Color.BLACK)
            val p = Paint().apply { color = Color.WHITE; style = Paint.Style.FILL }
            for (line in region.lines) {
                val q = line.quad
                if (q.size < 4) continue
                val path = android.graphics.Path().apply {
                    moveTo(q[0].x - x0, q[0].y - y0)
                    for (i in 1..3) lineTo(q[i].x - x0, q[i].y - y0)
                    close()
                }
                drawPath(path, p)
            }
        }
        val qm = IntArray(bw * bh)
        qmBmp.getPixels(qm, 0, bw, 0, 0, bw, bh)
        qmBmp.recycle()
        var n = 0; var sl = 0.0; var sl2 = 0.0; var sr = 0L; var sg = 0L; var sb = 0L
        for (y in 0 until bh) {
            for (x in 0 until bw) {
                if ((qm[y * bw + x] and 0xFF) <= 127) continue
                val gi = (y0 + y) * w + (x0 + x)
                if ((tightPx[gi] and 0xFF) > 127) continue // 文字像素
                val p = px[gi]
                val r = (p shr 16) and 0xFF; val g = (p shr 8) and 0xFF; val b = p and 0xFF
                val lum = 0.299 * r + 0.587 * g + 0.114 * b
                sl += lum; sl2 += lum * lum; sr += r; sg += g; sb += b; n++
            }
        }
        if (n < 16) return BgStat(255f, 0f, Color.WHITE)
        val mean = sl / n
        val std = kotlin.math.sqrt((sl2 / n - mean * mean).coerceAtLeast(0.0))
        // 眾數色：抗鋸齒字緣（不在 tight 遮罩內）會把均值拉灰；平塗改用最常見的顏色，並量它佔背景的比例判斷是否單色底。
        val bins = IntArray(32 * 32 * 32)
        for (y in 0 until bh) for (x in 0 until bw) {
            if ((qm[y * bw + x] and 0xFF) <= 127) continue
            val gi = (y0 + y) * w + (x0 + x)
            if ((tightPx[gi] and 0xFF) > 127) continue
            val p = px[gi]
            bins[(((p shr 19) and 31) shl 10) or (((p shr 11) and 31) shl 5) or ((p shr 3) and 31)]++
        }
        var best = 0
        for (i in bins.indices) if (bins[i] > bins[best]) best = i
        val br = ((best shr 10) and 31) * 8 + 4; val bg = ((best shr 5) and 31) * 8 + 4; val bb = (best and 31) * 8 + 4
        var near = 0; var mr = 0L; var mg = 0L; var mb = 0L
        for (y in 0 until bh) for (x in 0 until bw) {
            if ((qm[y * bw + x] and 0xFF) <= 127) continue
            val gi = (y0 + y) * w + (x0 + x)
            if ((tightPx[gi] and 0xFF) > 127) continue
            val p = px[gi]
            val r = (p shr 16) and 0xFF; val g = (p shr 8) and 0xFF; val b = p and 0xFF
            if (abs(r - br) <= MODE_TOL && abs(g - bg) <= MODE_TOL && abs(b - bb) <= MODE_TOL) {
                near++; mr += r; mg += g; mb += b
            }
        }
        val modeColor = if (near > 0) Color.rgb((mr / near).toInt(), (mg / near).toInt(), (mb / near).toInt()) else Color.rgb(br, bg, bb)
        return BgStat(
            mean.toFloat(), std.toFloat(), Color.rgb((sr / n).toInt(), (sg / n).toInt(), (sb / n).toInt()),
            modeColor, near.toFloat() / n,
        )
    }

    /** 白泡去字＝把區域 bbox(外擴 pad)內的去字遮罩像素直接平塗成背景色。均勻白泡保證無殘留。 */
    private fun flatFill(result: Bitmap, maskPx: IntArray, region: TextRegion, color: Int, pad: Int, w: Int, h: Int) {
        val x0 = (region.x0.toInt() - pad).coerceIn(0, w - 1)
        val y0 = (region.y0.toInt() - pad).coerceIn(0, h - 1)
        val x1 = (region.x1.toInt() + pad).coerceIn(x0 + 1, w)
        val y1 = (region.y1.toInt() + pad).coerceIn(y0 + 1, h)
        val bw = x1 - x0; val bh = y1 - y0
        val sub = IntArray(bw * bh)
        result.getPixels(sub, 0, bw, x0, y0, bw, bh)
        for (y in 0 until bh) {
            val mrow = (y0 + y) * w + x0
            val row = y * bw
            for (x in 0 until bw) if ((maskPx[mrow + x] and 0xFF) > 127) sub[row + x] = color
        }
        result.setPixels(sub, 0, bw, x0, y0, bw, bh)
    }

    private class WinOut(val x0: Int, val y0: Int, val ww: Int, val wh: Int, val px: IntArray)

    /**
     * 整頁跑一次 NCNN AOT-GAN：整張縮到方形 [cfg.tileSize] → 推論 → 放回原尺寸。只讀 page/maskBmp ⇒ 併發安全。
     * NCNN net 固定方形輸入、同尺寸復用安全（跨尺寸 reuse 會崩，故一律整頁 tileSize）。
     */
    private fun runWholeAot(page: Bitmap, maskBmp: Bitmap, w: Int, h: Int): WinOut? {
        val t = cfg.tileSize
        val imgScaled = Bitmap.createScaledBitmap(page, t, t, true)
        val maskScaled = Bitmap.createScaledBitmap(maskBmp, t, t, false)
        return try {
            val imgChw = aotImageChw(imgScaled, maskScaled, t)
            val maskArr = maskArr(maskScaled, t)
            val out = FloatArray(3 * t * t)
            val rc = NcnnBackend.inpaintAot(ncnnHandle, imgChw, maskArr, t, out)
            check(rc == 0) { "NCNN AOT 推論失敗 rc=$rc" }
            val resScaled = aotArrToBitmap(out, t)
            val resWin = Bitmap.createScaledBitmap(resScaled, w, h, true)
            val px = IntArray(w * h)
            resWin.getPixels(px, 0, w, 0, 0, w, h)
            if (resWin !== resScaled) resScaled.recycle()
            resWin.recycle()
            WinOut(0, 0, w, h, px)
        } catch (t2: Throwable) {
            Log.w(TAG, "AOT 去字失敗：${t2.message}"); null
        } finally {
            if (imgScaled !== page) imgScaled.recycle()
            if (maskScaled !== maskBmp) maskScaled.recycle()
        }
    }

    /** AOT 影像的裸 NCHW 陣列 [3*n*n]：RGB→[-1,1]，遮罩處歸零（m-i-t `img*(1-mask)`）。 */
    private fun aotImageChw(imgBmp: Bitmap, maskBmp: Bitmap, n: Int): FloatArray {
        val area = n * n
        val px = IntArray(area); imgBmp.getPixels(px, 0, n, 0, 0, n, n)
        val mp = IntArray(area); maskBmp.getPixels(mp, 0, n, 0, 0, n, n)
        val chw = FloatArray(3 * area)
        for (i in 0 until area) {
            if ((mp[i] and 0xFF) > 127) continue // 洞＝0
            val p = px[i]
            chw[i] = ((p shr 16) and 0xFF) / 127.5f - 1f
            chw[area + i] = ((p shr 8) and 0xFF) / 127.5f - 1f
            chw[2 * area + i] = (p and 0xFF) / 127.5f - 1f
        }
        return chw
    }

    /** 遮罩的裸陣列 [n*n]（1=擦）。 */
    private fun maskArr(bmp: Bitmap, n: Int): FloatArray {
        val px = IntArray(n * n); bmp.getPixels(px, 0, n, 0, 0, n, n)
        val m = FloatArray(n * n)
        for (i in px.indices) m[i] = if ((px[i] and 0xFF) > 127) 1f else 0f
        return m
    }

    /** AOT 輸出陣列 [3*n*n]∈[-1,1] → Bitmap（(x+1)*127.5）。 */
    private fun aotArrToBitmap(arr: FloatArray, n: Int): Bitmap {
        val area = n * n
        val px = IntArray(area)
        for (i in 0 until area) {
            val r = ((arr[i] + 1f) * 127.5f).toInt().coerceIn(0, 255)
            val g = ((arr[area + i] + 1f) * 127.5f).toInt().coerceIn(0, 255)
            val b = ((arr[2 * area + i] + 1f) * 127.5f).toInt().coerceIn(0, 255)
            px[i] = Color.rgb(r, g, b)
        }
        return Bitmap.createBitmap(px, n, n, Bitmap.Config.ARGB_8888)
    }

    /** 把 AOT 輸出貼回 result，只換遮罩內像素（序列呼叫、寫入安全）。 */
    private fun compositePixels(result: Bitmap, maskPx: IntArray, o: WinOut) {
        val w = result.width
        val cur = IntArray(o.ww * o.wh)
        result.getPixels(cur, 0, o.ww, o.x0, o.y0, o.ww, o.wh)
        for (y in 0 until o.wh) {
            val maskRow = (o.y0 + y) * w + o.x0
            val row = y * o.ww
            for (x in 0 until o.ww) if ((maskPx[maskRow + x] and 0xFF) > 127) cur[row + x] = o.px[row + x]
        }
        result.setPixels(cur, 0, o.ww, o.x0, o.y0, o.ww, o.wh)
    }

    override fun close() {
        if (ncnnHandle != 0L) { NcnnBackend.releaseNet(ncnnHandle); ncnnHandle = 0L }
    }

    companion object {
        private const val TAG = "Inpainter"
        /** 長邊 > 短邊×此值 視為長條頁（條漫），改走視窗去字（見 [inpaintTall]）。一般單頁約 1.4、跨頁約 1/1.4。 */
        private const val TALL_RATIO = 1.8f
        /** 區背景有此比例以上的像素落在眾數色 ±[MODE_TOL] 內 → 視為單色底（乾淨白泡／色框），平塗即可、不跑 AOT。 */
        private const val FLAT_FRAC = 0.85f
        private const val MODE_TOL = 16
    }
}
