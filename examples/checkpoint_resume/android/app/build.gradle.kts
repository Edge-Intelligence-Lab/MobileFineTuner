plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

val verifyContinuityInputs by tasks.registering {
    val nativeDir = project.file("src/main/jniLibs/arm64-v8a")
    val fixtureDir = project.file("src/main/assets/tiny_qwen")
    val required = listOf("libmft_owned_process.so", "libmft_qwen.so", "libmft_gemma.so")
        .map(nativeDir::resolve) +
        listOf("spec.json", "model/config.json", "model/model.safetensors", "train.jsonl", "valid.jsonl")
            .map(fixtureDir::resolve)
    doLast {
        val missing = required.filterNot { it.isFile && it.length() > 0 }
        check(missing.isEmpty()) {
            "Missing or empty APK inputs:\n" + missing.joinToString("\n") +
                "\nBuild both arm64 trainers and run python3 tools/make_android_fixture.py " +
                "from the example root (see ANDROID_CONTINUITY.md)."
        }
    }
}

tasks.named("preBuild") { dependsOn(verifyContinuityInputs) }

android {
    namespace = "com.mobilefinetuner.visualizer"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.mobilefinetuner.continuity"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "1.0.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary = true
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.14"
    }

    packaging {
        jniLibs { useLegacyPackaging = true }
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    implementation("com.google.android.material:material:1.12.0")

    implementation(platform("androidx.compose:compose-bom:2024.09.02"))
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.5")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.5")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.5")
    implementation("androidx.activity:activity-compose:1.9.2")
    implementation("androidx.documentfile:documentfile:1.0.1")
    implementation("androidx.work:work-runtime-ktx:2.9.1")

    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")

    testImplementation("junit:junit:4.13.2")
    testImplementation("com.google.truth:truth:1.4.4")

    androidTestImplementation(platform("androidx.compose:compose-bom:2024.09.02"))
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")

    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
}
