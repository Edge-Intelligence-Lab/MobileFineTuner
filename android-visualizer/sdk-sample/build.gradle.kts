plugins {
    id("com.android.application")
}

android {
    namespace = "com.mobilefinetuner.sdk.sample"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.mobilefinetuner.sdk.sample"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"

        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation(project(":mft-sdk"))
}
