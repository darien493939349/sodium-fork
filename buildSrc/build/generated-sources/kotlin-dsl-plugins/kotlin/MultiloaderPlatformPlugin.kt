/**
 * Precompiled [multiloader-platform.gradle.kts][Multiloader_platform_gradle] script plugin.
 *
 * @see Multiloader_platform_gradle
 */
public
class MultiloaderPlatformPlugin : org.gradle.api.Plugin<org.gradle.api.Project> {
    override fun apply(target: org.gradle.api.Project) {
        try {
            Class
                .forName("Multiloader_platform_gradle")
                .getDeclaredConstructor(org.gradle.api.Project::class.java, org.gradle.api.Project::class.java)
                .newInstance(target, target)
        } catch (e: java.lang.reflect.InvocationTargetException) {
            throw e.targetException
        }
    }
}
