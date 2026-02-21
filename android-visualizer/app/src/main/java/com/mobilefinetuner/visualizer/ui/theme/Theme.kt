package com.mobilefinetuner.visualizer.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

// ── iOS-inspired light color scheme (default) ─────────────────────────────────
private val LightColors = lightColorScheme(
    primary              = NeonBlue,                    // #007AFF
    onPrimary            = Color.White,
    primaryContainer     = Color(0xFFD6EAFF),           // light blue tint
    onPrimaryContainer   = Color(0xFF003380),
    secondary            = NeonGreen,                   // #34C759
    onSecondary          = Color.White,
    secondaryContainer   = Color(0xFFD4F5DC),           // light green tint
    onSecondaryContainer = Color(0xFF00461A),
    tertiary             = NeonPurple,                  // #AF52DE
    onTertiary           = Color.White,
    tertiaryContainer    = Color(0xFFF3E4FF),           // light purple tint
    onTertiaryContainer  = Color(0xFF520080),
    error                = NeonRed,                     // #FF3B30
    onError              = Color.White,
    errorContainer       = Color(0xFFFFECEB),
    onErrorContainer     = Color(0xFF800000),
    background           = GroupedBackground,           // #F2F2F7
    onBackground         = PrimaryLabel,                // #000000
    surface              = SystemBackground,            // #FFFFFF
    onSurface            = PrimaryLabel,                // #000000
    surfaceVariant       = SecondaryGrouped,            // #E5E5EA
    onSurfaceVariant     = SecondaryLabel,              // #8E8E93
    outline              = Separator,                   // #C6C6C8
    outlineVariant       = TertiaryGrouped              // #D1D1D6
)

// ── Dark fallback scheme (opt-in only) ────────────────────────────────────────
private val DarkColors = darkColorScheme(
    primary              = NeonBlue,
    onPrimary            = Color(0xFF001F5C),
    primaryContainer     = Color(0xFF00368A),
    onPrimaryContainer   = Color(0xFFD6E3FF),
    secondary            = NeonGreen,
    onSecondary          = Color(0xFF00391A),
    secondaryContainer   = Color(0xFF005227),
    onSecondaryContainer = Color(0xFF72F397),
    tertiary             = NeonPurple,
    onTertiary           = Color(0xFF3B0060),
    tertiaryContainer    = Color(0xFF560089),
    onTertiaryContainer  = Color(0xFFF0B8FF),
    error                = Color(0xFFFF6B62),
    onError              = Color(0xFF690005),
    background           = Color(0xFF1C1C1E),
    onBackground         = Color(0xFFE5E5EA),
    surface              = Color(0xFF2C2C2E),
    onSurface            = Color(0xFFE5E5EA),
    surfaceVariant       = Color(0xFF3A3A3C),
    onSurfaceVariant     = Color(0xFFAEAEB2),
    outline              = Color(0xFF636366),
    outlineVariant       = Color(0xFF48484A)
)

@Composable
fun MobileFineTunerVisualizerTheme(
    darkTheme: Boolean = false,   // iOS-style light theme by default
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = if (darkTheme) DarkColors else LightColors,
        typography  = AppTypography,
        content     = content
    )
}
