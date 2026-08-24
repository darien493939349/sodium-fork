package com.rustium;

import net.fabricmc.loader.api.FabricLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

public class RustLib {
    private static final Logger LOGGER = LoggerFactory.getLogger("Rustium");
    private static boolean loaded = false;

    static {
        try {
            loadNativeLibrary();
            loaded = true;
            LOGGER.info("[Rustium] Native library loaded successfully!");
            LOGGER.info("[Rustium] " + getRustDebugInfo());
        } catch (Exception e) {
            loaded = false;
            LOGGER.error("[Rustium] Failed to load native library!", e);
        }
    }

    public static boolean isLoaded() {
        return loaded;
    }

    private static void loadNativeLibrary() throws IOException {
        String osName = System.getProperty("os.name").toLowerCase();
        String osArch = System.getProperty("os.arch").toLowerCase();
        
        String libName;
        if (osName.contains("linux")) {
            libName = "libsodium_rust.so";
        } else if (osName.contains("mac")) {
            libName = "libsodium_rust.dylib";
        } else if (osName.contains("win")) {
            libName = "sodium_rust.dll";
        } else {
            throw new RuntimeException("Unsupported OS: " + osName);
        }

        // Path inside the JAR
        String resourcePath = "/natives/" + libName;
        
        // Extract to temp file
        Path tempFile = Files.createTempFile("rustium_", "_" + libName);
        tempFile.toFile().deleteOnExit();

        try (InputStream in = RustLib.class.getResourceAsStream(resourcePath);
             OutputStream out = Files.newOutputStream(tempFile)) {
            
            if (in == null) {
                throw new FileNotFoundException("Native library not found in JAR at " + resourcePath);
            }
            
            byte[] buffer = new byte[8192];
            int bytesRead;
            while ((bytesRead = in.read(buffer)) != -1) {
                out.write(buffer, 0, bytesRead);
            }
        }

        // Load the extracted library
        System.load(tempFile.toAbsolutePath().toString());
        LOGGER.info("[Rustium] Extracted and loaded {} from {}", libName, resourcePath);
    }

    // Native methods called by Java
    public static native String getRustDebugInfo();
    
    // Mesh Building
    public static native long buildChunkMeshParallel(byte[] blockData, long outputPtr, int stride, int count);
    
    // Frustum Culling
    public static native long testBoundsBatchParallel(float[] planes, float[] bounds);
    
    // Occlusion Culling
    public static native long createContext(int width, int height);
    public static native void destroyContext(long ctxPtr);
    public static native void buildHierarchyParallel(long ctxPtr, float[] viewProj, float[] chunks);
    public static native long testBatchParallel(long ctxPtr, float[] chunks);
    
    // Light Propagation
    public static native void propagateLightParallel(byte[] levels, long queuePtr, int queueCount);
    
    // Vertex Conversion
    public static native void convertVerticesBatchParallel(byte[] inputData, long outputPtr, int stride, int count);
}
