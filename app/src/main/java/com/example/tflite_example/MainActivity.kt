package com.example.tflite_example

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ThumbnailUtils
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.example.tflite_example.ml.ModelUnquant
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileNotFoundException
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.*
import kotlin.math.min


class MainActivity : AppCompatActivity() {
    
    private lateinit var result: TextView
    private lateinit var confidence: TextView
    private lateinit var imageView: ImageView
    private lateinit var picture: Button
    private var imageSize: Int = 224
    
//    private val capturePicture = registerForActivityResult(ActivityResultContracts.TakePicturePreview()) {
//        it?.let {
//            val dimension = min(it.width, it.height)
//
//            var image = ThumbnailUtils.extractThumbnail(it, dimension, dimension)
//            imageView.setImageBitmap(image)
//
//            image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false)
//            classifyImage(image)
//        }
//    }
    
    private val capturePicture = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) {
        it?.let {
            
            if (it.resultCode == Activity.RESULT_OK) {
                val selectedImage = it.data?.data ?: return@registerForActivityResult
                var imageStream: InputStream? = null
                try {
                    imageStream = contentResolver.openInputStream(selectedImage)
                } catch (e: FileNotFoundException) {
                    e.printStackTrace()
                }
                val bitmap = BitmapFactory.decodeStream(imageStream)
    
                val dimension = min(bitmap.width, bitmap.height)
    
                var image = ThumbnailUtils.extractThumbnail(bitmap, dimension, dimension)
                imageView.setImageBitmap(image)
    
                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false)
                classifyImage(image)
    
            }
            
        }
    }
    
    private fun classifyImage(image: Bitmap) {
        val model = ModelUnquant.newInstance(applicationContext)
        
        // Creates inputs for reference.
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
        val byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())
        
        val intValues = IntArray(imageSize * imageSize)
        image.getPixels(intValues, 0, image.width, 0, 0, image.width, image.height)
        
        var pixel = 0
        for (i in 0 until imageSize) {
            for (j in 0 until imageSize) {
                val value = intValues[pixel++]
                byteBuffer.putFloat(((value shr 16) and 0xFF) * (1f / 255f))
                byteBuffer.putFloat(((value shr 8) and 0xFF) * (1f / 255f))
                byteBuffer.putFloat((value and 0xFF) * (1f / 255f))
            }
        }
        
        inputFeature0.loadBuffer(byteBuffer)
        
        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
        
        val confidences = outputFeature0.floatArray
        val maxConfidence = confidences.max()
        val maxPos = confidences.indexOfFirst { it == maxConfidence }
        
        val classes = arrayOf("No cars", "Right", "Left")
        
        result.text = classes[maxPos]
        
        val s = buildString {
            classes.forEachIndexed { index, clas ->
                append(String.format("%s: %.1f%%\n", clas, confidences[index] * 100))
            }
        }
        confidence.text = s
        
        // Releases model resources if no longer used.
        model.close()
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        result = findViewById(R.id.result)
        confidence = findViewById(R.id.confidence)
        imageView = findViewById(R.id.imageView)
        picture = findViewById(R.id.button)
        
        picture.setOnClickListener {
            if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                val photoPickerIntent = Intent(Intent.ACTION_PICK)
                photoPickerIntent.type = "image/*"
                capturePicture.launch(photoPickerIntent)
            } else {
                requestPermissions(arrayOf(Manifest.permission.CAMERA), 100)
            }
        }
        
        
    }
}