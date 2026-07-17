package com.example.medicinalplantclassifier.ml

import android.util.Log

object ModelInspector {

    fun inspect(loader: ModelLoader) {

        val interpreter = loader.getInterpreter()

        val inputTensor = interpreter.getInputTensor(0)
        val outputTensor = interpreter.getOutputTensor(0)

        Log.d("MODEL_INFO", "==============================")
        Log.d("MODEL_INFO", "Input Shape : ${inputTensor.shape().contentToString()}")
        Log.d("MODEL_INFO", "Input Type  : ${inputTensor.dataType()}")

        Log.d("MODEL_INFO", "Output Shape: ${outputTensor.shape().contentToString()}")
        Log.d("MODEL_INFO", "Output Type : ${outputTensor.dataType()}")
        Log.d("MODEL_INFO", "==============================")
    }
}