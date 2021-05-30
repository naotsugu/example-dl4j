plugins {
    application
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.deeplearning4j:deeplearning4j-core:1.0.0-beta7")
    implementation("org.nd4j:nd4j-native-platform:1.0.0-beta7")
    implementation("org.apache.commons:commons-compress:1.20")
    implementation("org.slf4j:slf4j-jdk14:1.7.30")
}

application {
    mainClass.set("com.mammb.code.example.dl4j.App")
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(16))
    }
}
