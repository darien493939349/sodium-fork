plugins {
    id("multiloader-platform")

    id("net.fabricmc.fabric-loom") version ("1.16.1")
}

base {
    archivesName = "sodium-fabric"
}

val configurationApiModJava: Configuration = configurations.create("apiJava") {
    isCanBeResolved = true
}

val configurationCommonModJava: Configuration = configurations.create("commonJava") {
    isCanBeResolved = true
}

val configurationFrapiModJava: Configuration = configurations.create("frapiJava") {
    isCanBeResolved = true
}

val configurationApiModSources: Configuration = configurations.create("apiSources") {
    isCanBeResolved = true
}

val configurationCommonModResources: Configuration = configurations.create("commonResources") {
    isCanBeResolved = true
}

val configurationFrapiModResources: Configuration = configurations.create("frapiResources") {
    isCanBeResolved = true
}

dependencies {
    configurationCommonModJava(project(path = ":common", configuration = "commonMainJava"))
    configurationApiModJava(project(path = ":common", configuration = "commonApiJava"))
    configurationCommonModJava(project(path = ":common", configuration = "commonBootJava"))
    if (BuildConfig.SUPPORT_FRAPI) configurationFrapiModJava(project(path = ":frapi", configuration = "frapiMainJava"))

    configurationApiModSources(project(path = ":common", configuration = "commonApiSources"))

    configurationCommonModResources(project(path = ":common", configuration = "commonMainResources"))
    configurationCommonModResources(project(path = ":common", configuration = "commonApiResources"))
    configurationCommonModResources(project(path = ":common", configuration = "commonBootResources"))
    if (BuildConfig.SUPPORT_FRAPI) configurationFrapiModResources(project(path = ":frapi", configuration = "frapiMainResources"))
}

sourceSets.apply {
    main {
        compileClasspath += configurationCommonModJava
        compileClasspath += configurationApiModJava
        runtimeClasspath += configurationCommonModJava
        runtimeClasspath += configurationApiModJava
        if (BuildConfig.SUPPORT_FRAPI) {
            runtimeClasspath += configurationFrapiModJava
        }
    }
}

dependencies {
    minecraft("com.mojang:minecraft:${BuildConfig.MINECRAFT_VERSION}")

    implementation("net.fabricmc:fabric-loader:${BuildConfig.FABRIC_LOADER_VERSION}")

    fun addEmbeddedFabricModule(name: String) {
        val module = fabricApi.module(name, BuildConfig.FABRIC_API_VERSION)
        implementation(module)
        include(module)
    }

    // Fabric API modules
    addEmbeddedFabricModule("fabric-api-base")
    addEmbeddedFabricModule("fabric-block-getter-api-v2")
    addEmbeddedFabricModule("fabric-rendering-v1")

    if (BuildConfig.SUPPORT_FRAPI) {
        addEmbeddedFabricModule("fabric-renderer-api-v1")
    }

    addEmbeddedFabricModule("fabric-lifecycle-events-v1")
    addEmbeddedFabricModule("fabric-rendering-fluids-v1")
    addEmbeddedFabricModule("fabric-resource-loader-v0")
    addEmbeddedFabricModule("fabric-resource-loader-v1")
    addEmbeddedFabricModule("fabric-transitive-access-wideners-v1")
}

loom {
    accessWidenerPath.set(file("src/main/resources/sodium-fabric.accesswidener"))

    mixin {
        useLegacyMixinAp = false
    }

    runs {
        named("client") {
            client()
            configName = "Fabric/Client"
            appendProjectPathToConfigName = false
            ideConfigGenerated(true)
            runDir("run")
        }
    }
}

tasks {
    // Task to build Rust library and copy natives
    val buildRustNatives = register<Exec>("buildRustNatives") {
        workingDir(file("${rootProject.projectDir}/rust-sodium"))
        commandLine("cargo", "build", "--release")
        
        val rustTargetDir = file("${rootProject.projectDir}/rust-sodium/target/release")
        val nativesDir = file("${project.projectDir}/src/main/resources/natives")
        
        doLast {
            // Ensure natives directory exists
            nativesDir.mkdirs()
            
            // Copy Linux library
            val linuxLib = file("$rustTargetDir/libsodium_rust.so")
            if (linuxLib.exists()) {
                val destFile = file("$nativesDir/libsodium_rust.so")
                if (!destFile.exists() || linuxLib.length() != destFile.length()) {
                    copy {
                        from(linuxLib)
                        into(nativesDir)
                    }
                    println("[Rustium] Copied libsodium_rust.so to natives/")
                } else {
                    println("[Rustium] libsodium_rust.so already up-to-date")
                }
            }
            
            // Copy macOS library
            val macLib = file("$rustTargetDir/libsodium_rust.dylib")
            if (macLib.exists()) {
                val destFile = file("$nativesDir/libsodium_rust.dylib")
                if (!destFile.exists() || macLib.length() != destFile.length()) {
                    copy {
                        from(macLib)
                        into(nativesDir)
                    }
                    println("[Rustium] Copied libsodium_rust.dylib to natives/")
                } else {
                    println("[Rustium] libsodium_rust.dylib already up-to-date")
                }
            }
            
            // Copy Windows DLL
            val winLib = file("$rustTargetDir/sodium_rust.dll")
            if (winLib.exists()) {
                val destFile = file("$nativesDir/sodium_rust.dll")
                if (!destFile.exists() || winLib.length() != destFile.length()) {
                    copy {
                        from(winLib)
                        into(nativesDir)
                    }
                    println("[Rustium] Copied sodium_rust.dll to natives/")
                } else {
                    println("[Rustium] sodium_rust.dll already up-to-date")
                }
            }
        }
    }

    jar {
        from(configurationCommonModJava)
        from(configurationApiModJava)
        if (BuildConfig.SUPPORT_FRAPI) {
            from(configurationFrapiModJava)
        }
        // Include natives in the JAR
        from("src/main/resources/natives") {
            into("natives")
        }
    }

    val apiJar = register<org.gradle.jvm.tasks.Jar>("apiJar") {
        archiveClassifier.set("api")
        from(configurationApiModJava)
        from(sourceSets.main.get().resources)
        destinationDirectory.set(file(rootProject.layout.buildDirectory).resolve("api"))
    }

    val apiSourcesJar = register<org.gradle.jvm.tasks.Jar>("apiSourcesJar") {
        archiveClassifier.set("api-sources")
        from(configurationApiModSources)
        from(sourceSets.main.get().resources)
        destinationDirectory.set(file(rootProject.layout.buildDirectory).resolve("api-sources"))
    }

    jar {
        destinationDirectory.set(file(rootProject.layout.buildDirectory).resolve("mods"))
    }

    processResources {
        from(configurationCommonModResources)
        if (BuildConfig.SUPPORT_FRAPI) {
            from(configurationFrapiModResources)
        }
        // Ensure Rust natives are built before processing resources
        dependsOn("buildRustNatives")
    }
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            groupId = project.group as String
            artifactId = rootProject.name + "-" + project.name
            version = version

            from(components["java"])
        }

        create<MavenPublication>("mavenApi") {
            groupId = project.group as String
            artifactId = rootProject.name + "-" + project.name + "-api"
            version = version

            artifact(tasks.named("apiJar")) {
                classifier = null
            }

            artifact(tasks.named("apiSourcesJar")) {
                classifier = "sources"
            }

            pom.packaging = "jar"
        }
    }
}
