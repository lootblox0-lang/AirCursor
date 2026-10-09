package com.aircursor.app

import android.Manifest
import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PixelFormat
import android.graphics.Point
import android.hardware.display.DisplayManager
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.view.Display
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.view.accessibility.AccessibilityEvent
import android.widget.Toast
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.round
import kotlin.math.sqrt

// AirCursor: the front camera tracks your hand, a cursor follows the spot
// between your thumb and index tip. Pinching works like a real finger:
// pinch = finger down, move = drag, open = finger up.
class CursorService : AccessibilityService(), LifecycleOwner {

    // Tuning knobs, we tweak these after testing
    private val boxMin = 0.2f      // part of the camera view that maps to the screen edges
    private val boxMax = 0.8f
    private val smoothing = 0.35f  // lower = smoother but laggier
    private val pinchOn = 0.25f    // thumb/index gap (vs hand size) that counts as a pinch
    private val pinchOff = 0.40f   // gap needed to let go of the pinch
    private val slopDp = 12        // how far you move while pinched before it starts dragging
    private val moveMs = 30L       // length of each drag step sent to Android
    private val cursorDp = 28

    private val registry = LifecycleRegistry(this)
    override val lifecycle: Lifecycle get() = registry

    private val main = Handler(Looper.getMainLooper())
    private lateinit var cameraExecutor: ExecutorService
    private var cameraProvider: ProcessCameraProvider? = null
    private var analysis: ImageAnalysis? = null
    @Volatile private var tracker: HandLandmarker? = null
    @Volatile private var errorShown = false
    private var lastTs = 0L

    private lateinit var windowManager: WindowManager
    private lateinit var cursorParams: WindowManager.LayoutParams
    private var cursorView: CursorView? = null

    // Hand and cursor state
    private var curX = 0f
    private var curY = 0f
    private var hasPos = false
    private var pinching = false
    private var pinchFrames = 0
    private var missFrames = 0

    // Fake finger state
    private var stroke: GestureDescription.StrokeDescription? = null
    private var fingerX = 0f
    private var fingerY = 0f
    private var downX = 0f
    private var downY = 0f
    private var dragging = false
    private var gestureBusy = false
    private var liftPending = false
    private var liftX = 0f
    private var liftY = 0f
    private var sendId = 0

    private val displayListener = object : DisplayManager.DisplayListener {
        override fun onDisplayAdded(displayId: Int) {}
        override fun onDisplayRemoved(displayId: Int) {}
        override fun onDisplayChanged(displayId: Int) {
            if (displayId == Display.DEFAULT_DISPLAY) {
                analysis?.targetRotation = defaultDisplay().rotation
            }
        }
    }

    override fun onServiceConnected() {
        super.onServiceConnected()
        registry.currentState = Lifecycle.State.STARTED
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            toast("AirCursor: open the app and allow the camera first")
            return
        }
        addCursor()
        if (!setupTracker()) return
        cameraExecutor = Executors.newSingleThreadExecutor()
        startCamera()
        getSystemService(DisplayManager::class.java).registerDisplayListener(displayListener, main)
        toast("AirCursor on: show your hand to the camera")
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {}

    override fun onInterrupt() {}

    override fun onDestroy() {
        getSystemService(DisplayManager::class.java).unregisterDisplayListener(displayListener)
        cameraProvider?.unbindAll()
        if (::cameraExecutor.isInitialized) cameraExecutor.shutdown()
        tracker?.close()
        tracker = null
        cursorView?.let { windowManager.removeView(it) }
        cursorView = null
        if (registry.currentState.isAtLeast(Lifecycle.State.CREATED)) {
            registry.currentState = Lifecycle.State.DESTROYED
        }
        super.onDestroy()
    }

    private fun setupTracker(): Boolean = try {
        val base = BaseOptions.builder()
            .setModelAssetPath("hand_landmarker.task")
            .setDelegate(Delegate.CPU)
            .build()
        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setNumHands(1)
            .setMinHandDetectionConfidence(0.5f)
            .setMinHandPresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .setResultListener { result: HandLandmarkerResult, input: MPImage -> onResult(result, input) }
            .setErrorListener { e: RuntimeException -> showErrorOnce("tracker error: ${e.message}") }
            .build()
        tracker = HandLandmarker.createFromOptions(this, options)
        true
    } catch (e: Exception) {
        toast("AirCursor: tracker failed to load: ${e.message}")
        false
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val provider = future.get()
                val useCase = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .setTargetRotation(defaultDisplay().rotation)
                    .build()
                useCase.setAnalyzer(cameraExecutor) { image -> analyze(image) }
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, useCase)
                cameraProvider = provider
                analysis = useCase
            } catch (e: Exception) {
                toast("AirCursor: camera failed: ${e.message}")
            }
        }, mainExecutor)
    }

    // Runs on the camera thread for every frame
    private fun analyze(image: ImageProxy) {
        val t = tracker
        if (t == null) {
            image.close()
            return
        }
        val frame = try {
            image.toBitmap()
        } catch (e: Exception) {
            image.close()
            return
        }
        val rotation = image.imageInfo.rotationDegrees
        image.close()

        val matrix = Matrix().apply {
            postRotate(rotation.toFloat())
            postScale(-1f, 1f) // mirror it like a selfie
        }
        val upright = Bitmap.createBitmap(frame, 0, 0, frame.width, frame.height, matrix, false)
        val ts = max(SystemClock.uptimeMillis(), lastTs + 1)
        lastTs = ts
        try {
            t.detectAsync(BitmapImageBuilder(upright).build(), ts)
        } catch (e: Exception) {
            showErrorOnce("tracker error: ${e.message}")
        }
    }

    // Runs on the tracker thread with the hand landmarks
    private fun onResult(result: HandLandmarkerResult, input: MPImage) {
        val hands = result.landmarks()
        if (hands.isEmpty()) {
            main.post { onNoHand() }
            return
        }
        val pts = hands[0]
        val w = input.width.toFloat()
        val h = input.height.toFloat()
        fun gap(a: Int, b: Int): Float {
            val dx = (pts[a].x() - pts[b].x()) * w
            val dy = (pts[a].y() - pts[b].y()) * h
            return sqrt(dx * dx + dy * dy)
        }
        // 4 = thumb tip, 8 = index tip, 0 = wrist, 9 = middle finger knuckle
        val ratio = gap(4, 8) / max(gap(0, 9), 1f)
        val px = (pts[4].x() + pts[8].x()) / 2f
        val py = (pts[4].y() + pts[8].y()) / 2f
        main.post { onHand(px, py, ratio) }
    }

    private fun onHand(px: Float, py: Float, ratio: Float) {
        missFrames = 0
        val screen = screenSize()
        val tx = ((px - boxMin) / (boxMax - boxMin)).coerceIn(0f, 1f) * (screen.x - 1)
        val ty = ((py - boxMin) / (boxMax - boxMin)).coerceIn(0f, 1f) * (screen.y - 1)

        if (!hasPos) {
            curX = tx
            curY = ty
            hasPos = true
        } else {
            curX += (tx - curX) * smoothing
            curY += (ty - curY) * smoothing
        }

        if (!pinching) {
            pinchFrames = if (ratio < pinchOn) pinchFrames + 1 else 0
            if (pinchFrames >= 2) {
                pinching = true
                fingerDown(curX, curY)
            }
        } else if (ratio > pinchOff) {
            pinching = false
            pinchFrames = 0
            fingerUp(curX, curY)
        } else {
            fingerMove(curX, curY)
        }
        moveCursor(curX, curY, pinching)
    }

    private fun onNoHand() {
        missFrames++
        if (missFrames >= 5) {
            if (pinching) fingerUp(curX, curY)
            hasPos = false
            pinching = false
            pinchFrames = 0
            cursorView?.visibility = View.INVISIBLE
        }
    }

    // Finger down: press and keep holding
    private fun fingerDown(x: Float, y: Float) {
        val sx = round(x)
        val sy = round(y)
        val path = Path().apply { moveTo(sx, sy) }
        val s = GestureDescription.StrokeDescription(path, 0L, 10L, true)
        stroke = s
        fingerX = sx
        fingerY = sy
        downX = sx
        downY = sy
        dragging = false
        liftPending = false
        send(s)
    }

    // Finger moving: send the next bit of the drag, one step at a time
    private fun fingerMove(x: Float, y: Float) {
        val prev = stroke ?: return
        if (gestureBusy) return
        val nx = round(x)
        val ny = round(y)
        if (!dragging) {
            if (hypot(nx - downX, ny - downY) < slopDp * resources.displayMetrics.density) return
            dragging = true
        }
        if (nx == fingerX && ny == fingerY) return
        val path = Path().apply {
            moveTo(fingerX, fingerY)
            lineTo(nx, ny)
        }
        val s = prev.continueStroke(path, 0L, moveMs, true)
        stroke = s
        fingerX = nx
        fingerY = ny
        send(s)
    }

    // Finger up: let go (waits if a drag step is still running)
    private fun fingerUp(x: Float, y: Float) {
        if (stroke == null) return
        if (gestureBusy) {
            liftPending = true
            liftX = x
            liftY = y
            return
        }
        lift(x, y)
    }

    private fun lift(x: Float, y: Float) {
        val prev = stroke ?: return
        liftPending = false
        val nx = round(x)
        val ny = round(y)
        val moving = dragging && (nx != fingerX || ny != fingerY)
        val path = Path().apply {
            moveTo(fingerX, fingerY)
            if (moving) lineTo(nx, ny)
        }
        stroke = null
        send(prev.continueStroke(path, 0L, if (moving) moveMs else 10L, false))
    }

    private fun send(s: GestureDescription.StrokeDescription) {
        val id = ++sendId
        gestureBusy = true
        val callback = object : AccessibilityService.GestureResultCallback() {
            override fun onCompleted(gestureDescription: GestureDescription?) {
                if (id != sendId) return
                gestureBusy = false
                if (liftPending) lift(liftX, liftY)
            }

            override fun onCancelled(gestureDescription: GestureDescription?) {
                if (id != sendId) return
                gestureBusy = false
                liftPending = false
                stroke = null
            }
        }
        val gesture = GestureDescription.Builder().addStroke(s).build()
        if (!dispatchGesture(gesture, callback, main)) {
            gestureBusy = false
            stroke = null
        }
    }

    private fun addCursor() {
        windowManager = getSystemService(WINDOW_SERVICE) as WindowManager
        val size = (cursorDp * resources.displayMetrics.density).toInt()
        cursorParams = WindowManager.LayoutParams(
            size,
            size,
            WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
                WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.START
        }
        val view = CursorView(this)
        view.visibility = View.INVISIBLE
        windowManager.addView(view, cursorParams)
        cursorView = view
    }

    private fun moveCursor(x: Float, y: Float, pinched: Boolean) {
        val view = cursorView ?: return
        val half = cursorParams.width / 2
        cursorParams.x = x.toInt() - half
        cursorParams.y = y.toInt() - half
        view.pinched = pinched
        view.visibility = View.VISIBLE
        windowManager.updateViewLayout(view, cursorParams)
    }

    private fun defaultDisplay(): Display =
        getSystemService(DisplayManager::class.java).getDisplay(Display.DEFAULT_DISPLAY)

    @Suppress("DEPRECATION")
    private fun screenSize(): Point = Point().also { defaultDisplay().getRealSize(it) }

    private fun toast(msg: String) {
        main.post { Toast.makeText(this, msg, Toast.LENGTH_LONG).show() }
    }

    private fun showErrorOnce(msg: String) {
        if (errorShown) return
        errorShown = true
        toast("AirCursor: $msg")
    }

    // The little dot on screen: white normally, green while you pinch
    private class CursorView(context: Context) : View(context) {
        var pinched = false
            set(value) {
                if (field != value) {
                    field = value
                    invalidate()
                }
            }

        private val fill = Paint(Paint.ANTI_ALIAS_FLAG)
        private val ring = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            color = Color.BLACK
        }

        override fun onDraw(canvas: Canvas) {
            val c = width / 2f
            val r = c * 0.75f
            ring.strokeWidth = c * 0.2f
            fill.color = if (pinched) Color.rgb(0, 200, 120) else Color.argb(230, 255, 255, 255)
            canvas.drawCircle(c, c, r, fill)
            canvas.drawCircle(c, c, r, ring)
        }
    }
}