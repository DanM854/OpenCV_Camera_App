package com.example.opencv_camera_app

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import org.opencv.android.OpenCVLoader
import android.Manifest
import android.content.pm.PackageManager
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.CameraBridgeViewBase
import android.widget.Button
import android.widget.CheckBox
import android.widget.ImageView
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.android.Utils
import org.opencv.imgproc.Imgproc

class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {
    private lateinit var buttonStartPreview: Button
    private lateinit var buttonStopPreview: Button
    private lateinit var buttonSwitch: Button
    private lateinit var buttonGallery: Button
    private lateinit var checkBoxProcessing: CheckBox
    private lateinit var imageView: ImageView
    private lateinit var openCvCameraView: CameraBridgeViewBase
    private lateinit var inputMat: Mat
    private lateinit var processedMat: Mat

    private var isPreviewActive = false
    private lateinit var textViewStatus: TextView
    private var isOpenCvInitialized = false
    private val cameraPermissionRequestCode = 100
    private val galleryRequestCode = 200
    private var isUsingFrontCamera = false
    private var currentBitmap: Bitmap? = null // Imagen actual (cámara o galería)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        textViewStatus = findViewById(R.id.textViewStatus)
        buttonStartPreview = findViewById(R.id.buttonStartPreview)
        buttonStopPreview = findViewById(R.id.buttonStopPreview)
        buttonSwitch = findViewById(R.id.buttonSwitch)
        buttonGallery = findViewById(R.id.buttonGallery)
        checkBoxProcessing = findViewById(R.id.checkboxEnableProcessing)
        imageView = findViewById(R.id.imageView)
        openCvCameraView = findViewById(R.id.cameraView)

        isOpenCvInitialized = OpenCVLoader.initLocal()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                cameraPermissionRequestCode
            )
        }

        openCvCameraView.setCameraIndex(0)
        openCvCameraView.setCvCameraViewListener(this)

        buttonStartPreview.setOnClickListener {
            openCvCameraView.setCameraPermissionGranted()
            openCvCameraView.enableView()
            isPreviewActive = true
            currentBitmap = null
            updateControls()
        }

        buttonStopPreview.setOnClickListener {
            stopCamera()
        }

        buttonSwitch.setOnClickListener {
            switchCamera()
        }

        buttonGallery.setOnClickListener {
            openGallery()
        }

        checkBoxProcessing.setOnCheckedChangeListener { _, _ ->
            applyProcessingToCurrentImage()
        }

        updateControls()
    }

    private fun switchCamera() {
        if (!isOpenCvInitialized) return

        openCvCameraView.disableView()

        isUsingFrontCamera = !isUsingFrontCamera
        val newCameraIndex = if (isUsingFrontCamera) 1 else 0
        openCvCameraView.setCameraIndex(newCameraIndex)

        openCvCameraView.enableView()
    }

    private fun openGallery() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        startActivityForResult(intent, galleryRequestCode)
    }

    private fun stopCamera() {
        openCvCameraView.disableView()
        isPreviewActive = false
        updateControls()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == galleryRequestCode && resultCode == Activity.RESULT_OK && data != null) {
            val selectedImageUri: Uri? = data.data

            try {
                val bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, selectedImageUri)
                stopCamera() // Detiene la cámara automáticamente
                currentBitmap = bitmap
                applyProcessingToCurrentImage()
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    private fun applyProcessingToCurrentImage() {
        currentBitmap?.let { bitmap ->
            val mat = Mat()
            Utils.bitmapToMat(bitmap, mat)

            val processedBitmap = if (checkBoxProcessing.isChecked) {
                applyFilterToBitmap(mat)
            } else {
                bitmap
            }

            imageView.setImageBitmap(processedBitmap)
        }
    }

    private fun applyFilterToBitmap(mat: Mat): Bitmap {
        val processedMat = Mat()

        Imgproc.cvtColor(mat, processedMat, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.adaptiveThreshold(
            processedMat, processedMat, 255.0,
            Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
            Imgproc.THRESH_BINARY, 21, 0.0
        )

        val filteredBitmap = Bitmap.createBitmap(
            processedMat.cols(), processedMat.rows(), Bitmap.Config.ARGB_8888
        )
        Utils.matToBitmap(processedMat, filteredBitmap)

        return filteredBitmap
    }

    private fun updateControls() {
        if (!isOpenCvInitialized) {
            textViewStatus.text = "OpenCV initialization error"
            buttonStartPreview.isEnabled = false
            buttonStopPreview.isEnabled = false
        } else {
            textViewStatus.text = "OpenCV initialized"
            buttonStartPreview.isEnabled = !isPreviewActive
            buttonStopPreview.isEnabled = isPreviewActive
        }
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        isPreviewActive = true
        inputMat = Mat(height, width, CvType.CV_8UC4)
        processedMat = Mat(height, width, CvType.CV_8UC1)
        updateControls()
    }

    override fun onCameraViewStopped() {
        isPreviewActive = false
        inputMat.release()
        processedMat.release()
        updateControls()
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame?): Mat {
        inputFrame!!.rgba().copyTo(inputMat)

        var matToDisplay = inputMat
        if (checkBoxProcessing.isChecked) {
            matToDisplay = applyFilterToMat(inputMat)
        }

        val bitmapToDisplay = Bitmap.createBitmap(matToDisplay.cols(), matToDisplay.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(matToDisplay, bitmapToDisplay)

        runOnUiThread {
            imageView.setImageBitmap(bitmapToDisplay)
        }

        return inputMat
    }

    private fun applyFilterToMat(inputMat: Mat): Mat {
        val processedMat = Mat()
        Imgproc.cvtColor(inputMat, processedMat, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.adaptiveThreshold(
            processedMat, processedMat, 255.0,
            Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
            Imgproc.THRESH_BINARY, 21, 0.0
        )
        return processedMat
    }
}