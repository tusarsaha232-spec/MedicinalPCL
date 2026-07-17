package com.example.medicinalplantclassifier.ml

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import com.example.medicinalplantclassifier.data.Prediction

class PredictionEngine(
    private val context: Context,
    private val loader: ModelLoader
) {

    private val interpreter: Interpreter = loader.getInterpreter()
    private val labels: List<String> = loader.getLabels()

    fun predict(bitmap: Bitmap): Prediction {

        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

        val inputBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3)
        inputBuffer.order(ByteOrder.nativeOrder())

        for (y in 0 until 224) {
            for (x in 0 until 224) {

                val pixel = resized.getPixel(x, y)

//                val r = ((pixel shr 16) and 0xFF) / 255f
//                val g = ((pixel shr 8) and 0xFF) / 255f
//                val b = (pixel and 0xFF) / 255f
                  val r = (((pixel shr 16) and 0xFF) / 255f - 0.5f) / 0.5f
                  val g = (((pixel shr 8) and 0xFF) / 255f - 0.5f) / 0.5f
                  val b = (((pixel) and 0xFF) / 255f - 0.5f) / 0.5f

                inputBuffer.putFloat(r)
                inputBuffer.putFloat(g)
                inputBuffer.putFloat(b)
            }
        }

        val output = Array(1) { FloatArray(labels.size) }

        interpreter.run(inputBuffer, output)
        for (i in output[0].indices) {
            android.util.Log.d(
                "MODEL_OUTPUT",
                "${labels[i]} = ${output[0][i]}"
            )
        }

        var maxIndex = 0
        var maxConfidence = output[0][0]

        for (i in output[0].indices) {
            if (output[0][i] > maxConfidence) {
                maxConfidence = output[0][i]
                maxIndex = i
            }
        }

        return Prediction(
            label = labels[maxIndex],
            confidence = maxConfidence
        )
    }
}