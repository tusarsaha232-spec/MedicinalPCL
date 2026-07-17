package com.example.medicinalplantclassifier.ui

import android.app.Application
import android.graphics.Bitmap
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.medicinalplantclassifier.data.Prediction
import com.example.medicinalplantclassifier.ml.ModelInspector
import com.example.medicinalplantclassifier.ml.ModelLoader
import com.example.medicinalplantclassifier.ml.PredictionEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class PlantViewModel(application: Application) : AndroidViewModel(application) {

    private val modelLoader = ModelLoader(application)
    private val predictionEngine = PredictionEngine(application, modelLoader)

    init {
        ModelInspector.inspect(modelLoader)
    }

    var selectedBitmap by mutableStateOf<Bitmap?>(null)
        private set

    var result by mutableStateOf<Prediction?>(null)
        private set

    var isProcessing by mutableStateOf(false)
        private set

    fun onImageSelected(bitmap: Bitmap) {
        selectedBitmap = bitmap
        predict(bitmap)
    }

    private fun predict(bitmap: Bitmap) {
        viewModelScope.launch {
            isProcessing = true
            result = withContext(Dispatchers.Default) {
                predictionEngine.predict(bitmap)
            }
            isProcessing = false
        }
    }

    override fun onCleared() {
        super.onCleared()
        modelLoader.close()
    }
}
