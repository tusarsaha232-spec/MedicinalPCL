package com.example.medicinalplantclassifier.ml

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class ModelLoader(private val context: Context) {

    private val interpreter: Interpreter
    private val labels: List<String>

    init {
        interpreter = Interpreter(loadModelFile())
        labels = loadLabels()
    }

    private fun loadModelFile(): MappedByteBuffer {

        val fileDescriptor =
            context.assets.openFd("vectvmixer_float32.tflite")

        val inputStream =
            FileInputStream(fileDescriptor.fileDescriptor)

        val fileChannel =
            inputStream.channel

        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }

    private fun loadLabels(): List<String> {

        return context.assets
            .open("labels.txt")
            .bufferedReader()
            .readLines()
    }

    fun getInterpreter(): Interpreter {
        return interpreter
    }

    fun getLabels(): List<String> {
        return labels
    }

    fun close() {
        interpreter.close()
    }
}