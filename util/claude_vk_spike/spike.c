// claude_vk_spike: can a Vulkan compute shader write an image that OpenGL
// reads with NO COPY, on this machine, and what does the hand-off cost per
// frame? (DECISIONS 0v, the feasibility step before porting the tracer.)
//
//   GL side: EGL surfaceless context, GL 4.6 core, GL_EXT_memory_object_fd +
//            GL_EXT_semaphore_fd.
//   VK side: RADV compute, VK_KHR_external_memory_fd + _semaphore_fd.
//   One RGBA32F 1920x1080 image allocated by Vulkan, exported as an fd,
//   imported by GL. Two semaphores: vk_done (VK signals, GL waits) and
//   gl_done (GL signals, VK waits).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>
#include <GL/glext.h>

#define W 1920
#define H 1080
#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "FAIL %s = %d (line %d)\n", #x, r_, __LINE__); exit(1); } } while (0)
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

static PFNGLCREATEMEMORYOBJECTSEXTPROC glCreateMemoryObjectsEXT_;
static PFNGLIMPORTMEMORYFDEXTPROC glImportMemoryFdEXT_;
static PFNGLTEXTURESTORAGEMEM2DEXTPROC glTextureStorageMem2DEXT_;
static PFNGLGENSEMAPHORESEXTPROC glGenSemaphoresEXT_;
static PFNGLIMPORTSEMAPHOREFDEXTPROC glImportSemaphoreFdEXT_;
static PFNGLWAITSEMAPHOREEXTPROC glWaitSemaphoreEXT_;
static PFNGLSIGNALSEMAPHOREEXTPROC glSignalSemaphoreEXT_;
static PFNGLGETUNSIGNEDBYTEVEXTPROC glGetUnsignedBytevEXT_;
static PFNGLCREATETEXTURESPROC glCreateTextures_;
static PFNGLGETTEXTURESUBIMAGEPROC glGetTextureSubImage_;
static PFNGLGETSTRINGIPROC glGetStringi_;
static PFNGLMEMORYOBJECTPARAMETERIVEXTPROC glMemoryObjectParameterivEXT_;

static int has_gl_ext(const char *e) {
    GLint n = 0; glGetIntegerv(GL_NUM_EXTENSIONS, &n);
    for (int i = 0; i < n; i++) if (!strcmp((const char *)glGetStringi_(GL_EXTENSIONS, i), e)) return 1;
    return 0;
}
static unsigned char *read_file(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc(*n); fread(b, 1, *n, f); fclose(f); return b;
}

int main(int argc, char **argv) {
    int frames = argc > 1 ? atoi(argv[1]) : 600;
    // ---------------- GL: surfaceless EGL, 4.6 core
    PFNEGLGETPLATFORMDISPLAYEXTPROC getDisp = (void *)eglGetProcAddress("eglGetPlatformDisplayEXT");
    EGLDisplay dpy = getDisp(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
    EGLint maj, min;
    if (!eglInitialize(dpy, &maj, &min)) { fprintf(stderr, "FAIL eglInitialize\n"); return 1; }
    eglBindAPI(EGL_OPENGL_API);
    EGLint cfga[] = { EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE };
    EGLConfig cfg; EGLint ncfg = 0;
    eglChooseConfig(dpy, cfga, &cfg, 1, &ncfg);
    EGLint ctxa[] = { EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 6,
                      EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE };
    EGLContext ctx = eglCreateContext(dpy, ncfg ? cfg : EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ctxa);
    if (ctx == EGL_NO_CONTEXT || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) { fprintf(stderr, "FAIL GL 4.6 context\n"); return 1; }
#define LOAD(v, n) v = (void *)eglGetProcAddress(n); if (!v) { fprintf(stderr, "FAIL no %s\n", n); return 1; }
    LOAD(glGetStringi_, "glGetStringi");
    LOAD(glCreateMemoryObjectsEXT_, "glCreateMemoryObjectsEXT");
    LOAD(glImportMemoryFdEXT_, "glImportMemoryFdEXT");
    LOAD(glTextureStorageMem2DEXT_, "glTextureStorageMem2DEXT");
    LOAD(glGenSemaphoresEXT_, "glGenSemaphoresEXT");
    LOAD(glImportSemaphoreFdEXT_, "glImportSemaphoreFdEXT");
    LOAD(glWaitSemaphoreEXT_, "glWaitSemaphoreEXT");
    LOAD(glSignalSemaphoreEXT_, "glSignalSemaphoreEXT");
    LOAD(glGetUnsignedBytevEXT_, "glGetUnsignedBytevEXT");
    LOAD(glCreateTextures_, "glCreateTextures");
    LOAD(glGetTextureSubImage_, "glGetTextureSubImage");
    LOAD(glMemoryObjectParameterivEXT_, "glMemoryObjectParameterivEXT");
    printf("GL: %s | %s\n", glGetString(GL_VERSION), glGetString(GL_RENDERER));
    const char *need[] = { "GL_EXT_memory_object", "GL_EXT_memory_object_fd", "GL_EXT_semaphore", "GL_EXT_semaphore_fd" };
    for (int i = 0; i < 4; i++) { int h = has_gl_ext(need[i]); printf("  %-26s %s\n", need[i], h ? "yes" : "NO"); if (!h) return 1; }
    GLubyte gl_uuid[GL_UUID_SIZE_EXT];
    glGetUnsignedBytevEXT_(GL_DEVICE_UUID_EXT, gl_uuid);

    // ---------------- Vulkan: instance, the same GPU, compute queue
    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "claude_vk_spike", 1, NULL, 0, VK_API_VERSION_1_3 };
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app };
    VkInstance inst; CK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t npd = 8; VkPhysicalDevice pds[8]; CK(vkEnumeratePhysicalDevices(inst, &npd, pds));
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < npd; i++) {
        VkPhysicalDeviceIDProperties idp = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES };
        VkPhysicalDeviceProperties2 p2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &idp };
        vkGetPhysicalDeviceProperties2(pds[i], &p2);
        int same = !memcmp(idp.deviceUUID, gl_uuid, VK_UUID_SIZE);
        printf("VK device %u: %s  same GPU as GL: %s\n", i, p2.properties.deviceName, same ? "yes" : "no");
        if (same) pd = pds[i];
    }
    if (!pd) { fprintf(stderr, "FAIL no Vulkan device matches GL's UUID\n"); return 1; }
    uint32_t nq = 16; VkQueueFamilyProperties qf[16]; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf);
    uint32_t qfi = 0; for (uint32_t i = 0; i < nq; i++) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = i; break; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo dqci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, qfi, 1, &prio };
    const char *dexts[] = { VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME };
    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &dqci, 0, NULL, 2, dexts };
    VkDevice dev; CK(vkCreateDevice(pd, &dci, NULL, &dev));
    VkQueue q; vkGetDeviceQueue(dev, qfi, 0, &q);
    PFN_vkGetMemoryFdKHR getMemFd = (void *)vkGetDeviceProcAddr(dev, "vkGetMemoryFdKHR");
    PFN_vkGetSemaphoreFdKHR getSemFd = (void *)vkGetDeviceProcAddr(dev, "vkGetSemaphoreFdKHR");

    // ---------------- the shared image, allocated by Vulkan
    VkExternalMemoryImageCreateInfo emi = { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, NULL, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT };
    VkImageCreateInfo imci = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, &emi, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT,
        { W, H, 1 }, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED };
    VkImage img; CK(vkCreateImage(dev, &imci, NULL, &img));
    VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, img, &mr);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t mt = 0; for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((mr.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { mt = i; break; }
    VkMemoryDedicatedAllocateInfo ded = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, NULL, img, VK_NULL_HANDLE };
    VkExportMemoryAllocateInfo exa = { VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, &ded, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT };
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &exa, mr.size, mt };
    VkDeviceMemory mem; CK(vkAllocateMemory(dev, &mai, NULL, &mem));
    CK(vkBindImageMemory(dev, img, mem, 0));
    VkMemoryGetFdInfoKHR mgf = { VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, NULL, mem, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT };
    int memfd; CK(getMemFd(dev, &mgf, &memfd));

    // ---------------- two exportable semaphores
    VkExportSemaphoreCreateInfo esi = { VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO, NULL, VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT };
    VkSemaphoreCreateInfo sci = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &esi };
    VkSemaphore vk_done, gl_done; CK(vkCreateSemaphore(dev, &sci, NULL, &vk_done)); CK(vkCreateSemaphore(dev, &sci, NULL, &gl_done));
    int fd_vk, fd_gl;
    VkSemaphoreGetFdInfoKHR sgf = { VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR, NULL, vk_done, VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT };
    CK(getSemFd(dev, &sgf, &fd_vk)); sgf.semaphore = gl_done; CK(getSemFd(dev, &sgf, &fd_gl));

    // ---------------- GL imports the same memory and semaphores
    GLuint gmem; glCreateMemoryObjectsEXT_(1, &gmem);
    // Vulkan made a DEDICATED allocation for this one image; GL must be told,
    // or it guesses the tiling layout instead of reading it from the buffer
    GLint dedicated = getenv("SPIKE_NO_DEDICATED") ? GL_FALSE : GL_TRUE;
    glMemoryObjectParameterivEXT_(gmem, GL_DEDICATED_MEMORY_OBJECT_EXT, &dedicated);
    printf("GL memory object dedicated: %s\n", dedicated ? "yes" : "no");
    glImportMemoryFdEXT_(gmem, mr.size, GL_HANDLE_TYPE_OPAQUE_FD_EXT, memfd);
    GLuint tex; glCreateTextures_(GL_TEXTURE_2D, 1, &tex);
    glTextureStorageMem2DEXT_(tex, 1, GL_RGBA32F, W, H, gmem, 0);
    GLuint gs_vk, gs_gl; glGenSemaphoresEXT_(1, &gs_vk); glGenSemaphoresEXT_(1, &gs_gl);
    glImportSemaphoreFdEXT_(gs_vk, GL_HANDLE_TYPE_OPAQUE_FD_EXT, fd_vk);
    glImportSemaphoreFdEXT_(gs_gl, GL_HANDLE_TYPE_OPAQUE_FD_EXT, fd_gl);
    GLenum e = glGetError(); printf("GL import: %s (0x%x)\n", e == GL_NO_ERROR ? "ok" : "ERROR", e);
    if (e != GL_NO_ERROR) return 1;

    // ---------------- compute pipeline
    size_t spvn; unsigned char *spv = read_file(argc > 2 ? argv[2] : "fill.spv", &spvn);
    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, spvn, (uint32_t *)spv };
    VkShaderModule sm; CK(vkCreateShaderModule(dev, &smci, NULL, &sm));
    VkDescriptorSetLayoutBinding b = { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL };
    VkDescriptorSetLayoutCreateInfo dslci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 1, &b };
    VkDescriptorSetLayout dsl; CK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 4 };
    VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &dsl, 1, &pcr };
    VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
    VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL }, pl, VK_NULL_HANDLE, 0 };
    VkPipeline pipe; CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));
    VkImageViewCreateInfo ivci = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, img, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT,
        {0}, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
    VkImageView iv; CK(vkCreateImageView(dev, &ivci, NULL, &iv));
    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 };
    VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 1, &ps };
    VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
    VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 1, &dsl };
    VkDescriptorSet ds; CK(vkAllocateDescriptorSets(dev, &dsai, &ds));
    VkDescriptorImageInfo dii = { VK_NULL_HANDLE, iv, VK_IMAGE_LAYOUT_GENERAL };
    VkWriteDescriptorSet wds = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dii };
    vkUpdateDescriptorSets(dev, 1, &wds, 0, NULL);
    VkCommandPoolCreateInfo cpi = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qfi };
    VkCommandPool cp; CK(vkCreateCommandPool(dev, &cpi, NULL, &cp));
    VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
    VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &cbai, &cb));
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence; CK(vkCreateFence(dev, &fci, NULL, &fence));

    // ---------------- the loop: VK writes, GL waits and reads, GL hands back
    double t_vk = 0, t_gl = 0, t_all = 0;
    long bad = 0, checked = 0;
    float *row = malloc(sizeof(float) * 4 * W);
    for (int f = 0; f < frames; f++) {
        double t0 = now_ms();
        VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        CK(vkBeginCommandBuffer(cb, &bi));
        VkImageMemoryBarrier ib = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_SHADER_WRITE_BIT,
            f == 0 ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
            f == 0 ? VK_QUEUE_FAMILY_IGNORED : VK_QUEUE_FAMILY_EXTERNAL, f == 0 ? VK_QUEUE_FAMILY_IGNORED : qfi,
            img, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
        float fr = (float)f; vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &fr);
        vkCmdDispatch(cb, (W + 15) / 16, (H + 15) / 16, 1);
        // release to "external" (GL) for the read
        VkImageMemoryBarrier rb = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, 0,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, qfi, VK_QUEUE_FAMILY_EXTERNAL, img, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0, NULL, 1, &rb);
        CK(vkEndCommandBuffer(cb));
        VkPipelineStageFlags ws = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, f == 0 ? 0 : 1, &gl_done, &ws, 1, &cb, 1, &vk_done };
        CK(vkQueueSubmit(q, 1, &si, fence));
        double t1 = now_ms();
        // GL: wait for Vulkan on the GPU, then read one row and check it
        GLenum lay = GL_LAYOUT_GENERAL_EXT;
        glWaitSemaphoreEXT_(gs_vk, 0, NULL, 1, &tex, &lay);
        int y = (f * 97) % H;
        glGetTextureSubImage_(tex, 0, 0, y, 0, W, 1, 1, GL_RGBA, GL_FLOAT, sizeof(float) * 4 * W, row);
        for (int x = 0; x < W; x++) {
            checked++;
            if (row[4*x] != (float)x || row[4*x+1] != (float)y || row[4*x+2] != (float)f || row[4*x+3] != 1.0f) { if (bad < 4) printf("  frame %d x %d y %d: got (%g %g %g %g)\n", f, x, y, row[4*x], row[4*x+1], row[4*x+2], row[4*x+3]); bad++; }
        }
        glSignalSemaphoreEXT_(gs_gl, 0, NULL, 1, &tex, &lay);
        glFlush();
        double t2 = now_ms();
        CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX)); CK(vkResetFences(dev, 1, &fence));
        double t3 = now_ms();
        if (f >= 10) { t_vk += t1 - t0; t_gl += t2 - t1; t_all += t3 - t0; }
    }
    int n = frames - 10;
    printf("frames %d: pixels checked %ld, WRONG %ld\n", frames, checked, bad);
    printf("per frame (ms, after 10 warm-up): VK record+submit %.3f | GL wait+read row+signal %.3f | whole round trip %.3f\n",
           t_vk / n, t_gl / n, t_all / n);
    printf("%s\n", bad == 0 ? "RESULT: Vulkan-written image read by GL with no copy, every checked pixel exact"
                            : "RESULT: FAIL, GL read different values than Vulkan wrote");
    return bad != 0;
}
