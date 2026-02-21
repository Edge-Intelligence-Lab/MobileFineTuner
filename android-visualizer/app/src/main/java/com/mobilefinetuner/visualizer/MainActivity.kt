package com.mobilefinetuner.visualizer

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import com.mobilefinetuner.visualizer.ui.DashboardScreen
import com.mobilefinetuner.visualizer.ui.theme.MobileFineTunerVisualizerTheme
import com.mobilefinetuner.visualizer.viewmodel.ExperimentViewModel

class MainActivity : ComponentActivity() {

    private val viewModel by viewModels<ExperimentViewModel>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MobileFineTunerVisualizerTheme(darkTheme = false) {
                DashboardScreen(viewModel = viewModel)
            }
        }
    }
}
