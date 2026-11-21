package com.mobilefinetuner.visualizer.ui.theme

import androidx.compose.ui.graphics.Color

// ── iOS System Backgrounds ─────────────────────────────────────────────────────
val SystemBackground    = Color(0xFFFFFFFF)   // primary surface (cards)
val GroupedBackground   = Color(0xFFF2F2F7)   // app background (iOS grouped)
val SecondaryGrouped    = Color(0xFFE5E5EA)   // elevated / input fills
val TertiaryGrouped     = Color(0xFFD1D1D6)   // deepest inset
val Separator           = Color(0xFFC6C6C8)   // iOS separator

// ── iOS Labels ────────────────────────────────────────────────────────────────
val PrimaryLabel        = Color(0xFF000000)   // main text
val SecondaryLabel      = Color(0xFF8E8E93)   // secondary / caption
val TertiaryLabel       = Color(0xFFC7C7CC)   // placeholder / disabled

// ── iOS System Colors (chart series, semantic accents) ────────────────────────
val LossColor   = Color(0xFFFF3B30)   // systemRed    — Loss curve
val PplColor    = Color(0xFF007AFF)   // systemBlue   — PPL curve
val LrColor     = Color(0xFF34C759)   // systemGreen  — LR curve
val RssColor    = Color(0xFFAF52DE)   // systemPurple — RSS curve

// ── Accent palette ────────────────────────────────────────────────────────────
val NeonBlue    = Color(0xFF007AFF)   // systemBlue
val NeonGreen   = Color(0xFF34C759)   // systemGreen
val NeonAmber   = Color(0xFFFF9500)   // systemOrange
val NeonPurple  = Color(0xFFAF52DE)   // systemPurple
val NeonRed     = Color(0xFFFF3B30)   // systemRed
val NeonIndigo  = Color(0xFF5856D6)   // systemIndigo
val NeonTeal    = Color(0xFF5AC8FA)   // systemTeal

// ── Dynamic phase chart palette tokens ─────────────────────────────────────────
val WfBaseline  = Color(0xFF8E8E93)   // neutral
val WfFp16      = Color(0xFF007AFF)   // blue
val WfActCkpt   = Color(0xFF34C759)   // green
val WfMemAttn   = Color(0xFF30D158)   // green alt
val WfSharding  = Color(0xFFFF9500)   // orange
val WfFullOpt   = Color(0xFF5856D6)   // indigo

// ── Legacy tokens (kept for LightColors mapping, do not remove) ──────────────
val Clay            = Color(0xFF007AFF)
val ClayDark        = Color(0xFF0055CC)
val Moss            = Color(0xFF34C759)
val Sky             = Color(0xFF5AC8FA)
val Gold            = Color(0xFFFF9500)
val ErrorRed        = Color(0xFFFF3B30)
val SurfaceWarm     = Color(0xFFF2F2F7)
val SurfaceElevated = Color(0xFFFFFFFF)
