package com.example.opencv_camera_app

import android.os.Bundle
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import org.opencv.android.OpenCVLoader

class MainActivity : AppCompatActivity() {
    private lateinit var textViewStatus: TextView
    private var isOpenCvInitialized = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        textViewStatus = findViewById(R.id.textViewStatus)

        isOpenCvInitialized = OpenCVLoader.initLocal()

        updateControls()
    }

    private fun updateControls() {
        if(!isOpenCvInitialized){
            textViewStatus.text = "OpenCV initialization error"
        }else{
            textViewStatus.text = "OpenCV initialized"
        }
    }
}