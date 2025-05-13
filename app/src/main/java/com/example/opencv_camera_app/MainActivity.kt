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
import android.content.Context
import android.content.pm.PackageManager
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.CameraBridgeViewBase
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.ImageView
import android.widget.Spinner
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfRect
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.android.Utils
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

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
    private lateinit var spinnerFilters: Spinner

    private val FILTER_ORIGINAL = 0
    private val FILTER_ADAPTIVE_THRESHOLD = 1
    private val FILTER_NEGATIVE = 2
    private val FILTER_RED_ONLY = 3
    private val FILTER_FACE_DETECTION = 4
    private var currentFilter = FILTER_ORIGINAL

    private var isPreviewActive = false
    private lateinit var textViewStatus: TextView
    private var isOpenCvInitialized = false
    private val cameraPermissionRequestCode = 100
    private val galleryRequestCode = 200
    private var isUsingFrontCamera = false
    private var currentBitmap: Bitmap? = null // Imagen actual (cámara o galería)

    // Face detection variables
    private lateinit var faceCascade: CascadeClassifier
    private var faceCascadeLoaded = false
    private var frameCount = 0
    private val PROCESS_EVERY_N_FRAMES = 3 // Procesar cada 5 frames
    private var lastDetectedFaces = arrayOf<Rect>()
    private val isProcessingFaces = AtomicBoolean(false)
    private val faceDetectionExecutor = Executors.newSingleThreadExecutor()

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
        spinnerFilters = findViewById(R.id.spinnerFilters)

        // Configura el adaptador para el Spinner
        ArrayAdapter.createFromResource(
            this,
            R.array.filter_options,
            android.R.layout.simple_spinner_item
        ).also { adapter ->
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            spinnerFilters.adapter = adapter
        }

        // Configura el listener para el Spinner
        spinnerFilters.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>, view: View?, position: Int, id: Long) {
                currentFilter = position
                applyProcessingToCurrentImage()
            }

            override fun onNothingSelected(parent: AdapterView<*>) {
                // No hacer nada
            }
        }

        isOpenCvInitialized = OpenCVLoader.initLocal()

        // Initialize face detection
        initFaceDetection()

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

    private fun initFaceDetection() {
        try {
            // Load the cascade classifier from raw resource
            val inputStream = resources.openRawResource(R.raw.haarcascade_frontalface_alt)
            val cascadeDir = getDir("cascade", Context.MODE_PRIVATE)
            val cascadeFile = File(cascadeDir, "haarcascade_frontalface_alt.xml")
            val outStream = FileOutputStream(cascadeFile)

            val buffer = ByteArray(4096)
            var bytesRead: Int
            while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                outStream.write(buffer, 0, bytesRead)
            }

            inputStream.close()
            outStream.close()

            faceCascade = CascadeClassifier(cascadeFile.absolutePath)
            faceCascadeLoaded = !faceCascade.empty()

            if (!faceCascadeLoaded) {
                textViewStatus.text = "Failed to load face cascade classifier"
            } else {
                textViewStatus.text = "Face cascade classifier loaded successfully"
            }

            cascadeDir.delete()

        } catch (e: IOException) {
            e.printStackTrace()
            textViewStatus.text = "Failed to load face cascade classifier: ${e.message}"
        }
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

        when (currentFilter) {
            FILTER_ORIGINAL -> {
                mat.copyTo(processedMat)
            }
            FILTER_ADAPTIVE_THRESHOLD -> {
                Imgproc.cvtColor(mat, processedMat, Imgproc.COLOR_RGBA2GRAY)
                Imgproc.adaptiveThreshold(
                    processedMat, processedMat, 255.0,
                    Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                    Imgproc.THRESH_BINARY, 21, 0.0
                )
            }
            FILTER_NEGATIVE -> {
                mat.copyTo(processedMat)
                Core.bitwise_not(processedMat, processedMat)
            }
            FILTER_RED_ONLY -> {
                try {
                    // Asegurarse de que estamos trabajando con BGR
                    val bgrMat = Mat()
                    if (mat.channels() == 4) {
                        Imgproc.cvtColor(mat, bgrMat, Imgproc.COLOR_RGBA2BGR)
                    } else {
                        mat.copyTo(bgrMat)
                    }

                    // Convertir a HSV
                    val hsvMat = Mat()
                    Imgproc.cvtColor(bgrMat, hsvMat, Imgproc.COLOR_BGR2HSV)

                    // Definir rangos para el color rojo (en HSV)
                    val lowerRed1 = Scalar(0.0, 100.0, 100.0)
                    val upperRed1 = Scalar(10.0, 255.0, 255.0)
                    val lowerRed2 = Scalar(160.0, 100.0, 100.0)
                    val upperRed2 = Scalar(179.0, 255.0, 255.0)

                    // Crear máscaras
                    val mask1 = Mat()
                    val mask2 = Mat()
                    Core.inRange(hsvMat, lowerRed1, upperRed1, mask1)
                    Core.inRange(hsvMat, lowerRed2, upperRed2, mask2)

                    // Combinar máscaras
                    val redMask = Mat()
                    Core.addWeighted(mask1, 1.0, mask2, 1.0, 0.0, redMask)

                    // Convertir a escala de grises
                    val grayMat = Mat()
                    Imgproc.cvtColor(bgrMat, grayMat, Imgproc.COLOR_BGR2GRAY)

                    // Convertir de nuevo a color (pero gris)
                    val grayBGR = Mat()
                    Imgproc.cvtColor(grayMat, grayBGR, Imgproc.COLOR_GRAY2BGR)

                    // Crear una máscara inversa
                    val invertedMask = Mat()
                    Core.bitwise_not(redMask, invertedMask)

                    // Extraer solo la parte roja
                    val redPart = Mat()
                    bgrMat.copyTo(redPart, redMask)

                    // Extraer la parte gris
                    val grayPart = Mat()
                    grayBGR.copyTo(grayPart, invertedMask)

                    // Combinar ambas partes
                    processedMat.create(bgrMat.rows(), bgrMat.cols(), bgrMat.type())
                    Core.add(redPart, grayPart, processedMat)

                    // Si la entrada era RGBA, convertir la salida a RGBA también
                    if (mat.channels() == 4) {
                        Imgproc.cvtColor(processedMat, processedMat, Imgproc.COLOR_BGR2RGBA)
                    }

                } catch (e: Exception) {
                    // En caso de error, devolver la imagen original
                    mat.copyTo(processedMat)
                    e.printStackTrace()
                }
            }
            FILTER_FACE_DETECTION -> {
                mat.copyTo(processedMat)

                // Solo procesamos si el clasificador se cargó correctamente
                if (faceCascadeLoaded) {
                    detectFacesAndDraw(mat, processedMat)
                }
            }
        }

        val filteredBitmap = Bitmap.createBitmap(
            processedMat.cols(), processedMat.rows(), Bitmap.Config.ARGB_8888
        )
        Utils.matToBitmap(processedMat, filteredBitmap)

        return filteredBitmap
    }

    private fun detectFacesAndDraw(inputMat: Mat, outputMat: Mat) {
        // Convertir a escala de grises para detección
        val grayMat = Mat()
        Imgproc.cvtColor(inputMat, grayMat, Imgproc.COLOR_RGBA2GRAY)

        // Reducir el tamaño para procesamiento más rápido
        val smallGrayMat = Mat()
        val scale = 0.5 // Escalar a la mitad del tamaño
        val smallSize = Size(grayMat.cols() * scale, grayMat.rows() * scale)
        Imgproc.resize(grayMat, smallGrayMat, smallSize)

        // Detectar rostros
        val faces = MatOfRect()
        faceCascade.detectMultiScale(
            smallGrayMat,
            faces,
            1.2, // Factor de escala mayor para procesamiento más rápido
            3,
            0,
            Size(30.0 * scale, 30.0 * scale), // Ajustar tamaño mínimo
            Size() // Sin límite máximo
        )

        // Ajustar las coordenadas de las detecciones al tamaño original
        val facesArray = faces.toArray()
        val scaledFaces = facesArray.map { face ->
            Rect(
                (face.x / scale).toInt(),
                (face.y / scale).toInt(),
                (face.width / scale).toInt(),
                (face.height / scale).toInt()
            )
        }.toTypedArray()

        // Dibujar rectángulos verdes alrededor de los rostros
        for (face in scaledFaces) {
            Imgproc.rectangle(
                outputMat,
                Point(face.x.toDouble(), face.y.toDouble()),
                Point((face.x + face.width).toDouble(), (face.y + face.height).toDouble()),
                Scalar(0.0, 255.0, 0.0),  // Color verde
                3  // Grosor de la línea
            )
        }

        // Liberar memoria
        grayMat.release()
        smallGrayMat.release()
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
            matToDisplay = applyFilterToMatOptimized(inputMat)
        }

        val bitmapToDisplay = Bitmap.createBitmap(matToDisplay.cols(), matToDisplay.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(matToDisplay, bitmapToDisplay)

        runOnUiThread {
            imageView.setImageBitmap(bitmapToDisplay)
        }

        return inputMat
    }

    private fun applyFilterToMatOptimized(inputMat: Mat): Mat {
        val processedMat = Mat()

        when (currentFilter) {
            FILTER_ORIGINAL -> {
                inputMat.copyTo(processedMat)
            }
            FILTER_ADAPTIVE_THRESHOLD -> {
                Imgproc.cvtColor(inputMat, processedMat, Imgproc.COLOR_RGBA2GRAY)
                Imgproc.adaptiveThreshold(
                    processedMat, processedMat, 255.0,
                    Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                    Imgproc.THRESH_BINARY, 21, 0.0
                )
            }
            FILTER_NEGATIVE -> {
                inputMat.copyTo(processedMat)
                Core.bitwise_not(processedMat, processedMat)
            }
            FILTER_RED_ONLY -> {
                try {
                    // Asegurarse de que estamos trabajando con BGR
                    val bgrMat = Mat()
                    if (inputMat.channels() == 4) {
                        Imgproc.cvtColor(inputMat, bgrMat, Imgproc.COLOR_RGBA2BGR)
                    } else {
                        inputMat.copyTo(bgrMat)
                    }

                    // Convertir a HSV
                    val hsvMat = Mat()
                    Imgproc.cvtColor(bgrMat, hsvMat, Imgproc.COLOR_BGR2HSV)

                    // Definir rangos para el color rojo (en HSV)
                    val lowerRed1 = Scalar(0.0, 100.0, 100.0)
                    val upperRed1 = Scalar(10.0, 255.0, 255.0)
                    val lowerRed2 = Scalar(160.0, 100.0, 100.0)
                    val upperRed2 = Scalar(179.0, 255.0, 255.0)

                    // Crear máscaras
                    val mask1 = Mat()
                    val mask2 = Mat()
                    Core.inRange(hsvMat, lowerRed1, upperRed1, mask1)
                    Core.inRange(hsvMat, lowerRed2, upperRed2, mask2)

                    // Combinar máscaras
                    val redMask = Mat()
                    Core.addWeighted(mask1, 1.0, mask2, 1.0, 0.0, redMask)

                    // Convertir a escala de grises
                    val grayMat = Mat()
                    Imgproc.cvtColor(bgrMat, grayMat, Imgproc.COLOR_BGR2GRAY)

                    // Convertir de nuevo a color (pero gris)
                    val grayBGR = Mat()
                    Imgproc.cvtColor(grayMat, grayBGR, Imgproc.COLOR_GRAY2BGR)

                    // Crear una máscara inversa
                    val invertedMask = Mat()
                    Core.bitwise_not(redMask, invertedMask)

                    // Extraer solo la parte roja
                    val redPart = Mat()
                    bgrMat.copyTo(redPart, redMask)

                    // Extraer la parte gris
                    val grayPart = Mat()
                    grayBGR.copyTo(grayPart, invertedMask)

                    // Combinar ambas partes
                    processedMat.create(bgrMat.rows(), bgrMat.cols(), bgrMat.type())
                    Core.add(redPart, grayPart, processedMat)

                    // Si la entrada era RGBA, convertir la salida a RGBA también
                    if (inputMat.channels() == 4) {
                        Imgproc.cvtColor(processedMat, processedMat, Imgproc.COLOR_BGR2RGBA)
                    }

                } catch (e: Exception) {
                    // En caso de error, devolver la imagen original
                    inputMat.copyTo(processedMat)
                    e.printStackTrace()
                }
            }
            FILTER_FACE_DETECTION -> {
                inputMat.copyTo(processedMat)

                // Solo procesamos si el clasificador se cargó correctamente
                if (faceCascadeLoaded) {
                    // Optimización: Detectar caras solo en algunos frames
                    frameCount = (frameCount + 1) % PROCESS_EVERY_N_FRAMES

                    if (frameCount == 0 && !isProcessingFaces.get()) {
                        // Iniciar detección en un hilo separado
                        isProcessingFaces.set(true)
                        val inputCopy = inputMat.clone()

                        faceDetectionExecutor.execute {
                            try {
                                // Convertir a escala de grises para detección
                                val grayMat = Mat()
                                Imgproc.cvtColor(inputCopy, grayMat, Imgproc.COLOR_RGBA2GRAY)

                                // Reducir el tamaño para procesamiento más rápido
                                val smallGrayMat = Mat()
                                val scale = 0.5 // Escalar a la mitad del tamaño
                                val smallSize = Size(grayMat.cols() * scale, grayMat.rows() * scale)
                                Imgproc.resize(grayMat, smallGrayMat, smallSize)

                                // Detectar rostros
                                val faces = MatOfRect()
                                faceCascade.detectMultiScale(
                                    smallGrayMat,
                                    faces,
                                    1.2, // Factor de escala mayor para procesamiento más rápido
                                    3,
                                    0,
                                    Size(30.0 * scale, 30.0 * scale), // Ajustar tamaño mínimo
                                    Size() // Sin límite máximo
                                )

                                // Ajustar las coordenadas de las detecciones al tamaño original
                                val facesArray = faces.toArray()
                                lastDetectedFaces = facesArray.map { face ->
                                    Rect(
                                        (face.x / scale).toInt(),
                                        (face.y / scale).toInt(),
                                        (face.width / scale).toInt(),
                                        (face.height / scale).toInt()
                                    )
                                }.toTypedArray()

                                // Liberar memoria
                                grayMat.release()
                                smallGrayMat.release()
                                inputCopy.release()
                            } catch (e: Exception) {
                                e.printStackTrace()
                            } finally {
                                isProcessingFaces.set(false)
                            }
                        }
                    }

                    // Dibujar rectángulos verdes alrededor de los rostros
                    for (face in lastDetectedFaces) {
                        Imgproc.rectangle(
                            processedMat,
                            Point(face.x.toDouble(), face.y.toDouble()),
                            Point((face.x + face.width).toDouble(), (face.y + face.height).toDouble()),
                            Scalar(0.0, 255.0, 0.0),  // Color verde
                            3  // Grosor de la línea
                        )
                    }
                }
            }
        }

        return processedMat
    }

    override fun onDestroy() {
        super.onDestroy()
        faceDetectionExecutor.shutdown()
    }
}