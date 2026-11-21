#include <jni.h>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <new>
#include <pthread.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {
void fail(JNIEnv* env, const char* operation, int err) {
    char msg[256];
    snprintf(msg, sizeof(msg), "%s: %s (%d)", operation, strerror(err), err);
    env->ThrowNew(env->FindClass("java/io/IOException"), msg);
}

struct Strings {
    std::vector<char> bytes;
    std::vector<char*> pointers;
    bool load(JNIEnv* env, jbyteArray array) {
        bytes.resize(env->GetArrayLength(array));
        if (bytes.empty()) return false;
        env->GetByteArrayRegion(array, 0, bytes.size(), reinterpret_cast<jbyte*>(bytes.data()));
        if (env->ExceptionCheck() || bytes.back() != '\0') return false;
        pointers.push_back(bytes.data());
        for (size_t i = 1; i < bytes.size(); ++i)
            if (bytes[i-1] == '\0') pointers.push_back(bytes.data() + i);
        pointers.push_back(nullptr);
        return true;
    }
};

// Raw close in the fork child: inherited ART descriptors may carry fdsan owner tags.
void close_copy(int fd) { syscall(SYS_close, fd); }

bool close_inherited() {
    int dir = open("/proc/self/fd", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dir < 0) return false;
    struct Entry { uint64_t ino; int64_t off; unsigned short len; unsigned char type; char name[1]; };
    alignas(uint64_t) char buf[4096];
    for (;;) {
        long count = syscall(SYS_getdents64, dir, buf, sizeof(buf));
        if (count == 0) break;
        if (count < 0) return false;
        for (long pos = 0; pos < count;) {
            const auto* e = reinterpret_cast<const Entry*>(buf + pos);
            if (e->name[0] >= '0' && e->name[0] <= '9') {
                int fd = 0;
                for (const char* c = e->name; *c; ++c) fd = fd * 10 + (*c - '0');
                if (fd > 2 && fd != dir) close_copy(fd);
            }
            pos += e->len;
        }
    }
    close_copy(dir);
    return true;
}

// No allocator, JNI, logging framework, C++ cleanup, or parent acknowledgement after fork.
[[noreturn]] void exec_child(pid_t parent, char* const* spec, char* const* envp) {
    if (prctl(PR_SET_PDEATHSIG, SIGKILL, 0L, 0L, 0L) < 0 || getppid() != parent) _exit(126);
    // Close inherited descriptors before any potentially blocking child setup (including log open).
    // Reserve stdio first so /proc/self/fd never occupies one of its final slots.
    for (int i = 0; i < 3; ++i) close_copy(i);
    if (open("/dev/null", O_RDONLY) != 0 || open("/dev/null", O_WRONLY) != 1 || dup2(1, 2) != 2 ||
        !close_inherited()) _exit(126);
    int log = open(spec[1], O_WRONLY | O_APPEND | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (log < 0 || dup2(log, 1) < 0 || dup2(log, 2) < 0) _exit(126);
    close_copy(log);
    if (chdir(spec[0]) != 0) {
        constexpr char error[] = "trainer launch: chdir failed\n";
        write(2, error, sizeof(error)-1);
        _exit(126);
    }
    struct sigaction def{};
    def.sa_handler = SIG_DFL;
    sigemptyset(&def.sa_mask);
    for (int s = 1; s < NSIG; ++s)
        if (s != SIGKILL && s != SIGSTOP && sigaction(s, &def, nullptr) < 0 && errno != EINVAL) _exit(126);
    sigset_t clear;
    sigemptyset(&clear);
    if (sigprocmask(SIG_SETMASK, &clear, nullptr) < 0) _exit(126);
    execve(spec[2], spec + 2, envp);
    constexpr char error[] = "trainer launch: execve failed\n";
    write(2, error, sizeof(error)-1);
    _exit(127);
}

pid_t wait_child(pid_t pid, int* status, int flags) {
    pid_t result;
    do { result = waitpid(pid, status, flags); } while (result < 0 && errno == EINTR);
    return result;
}
int exit_code(int status) {
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return -1;
}
}

extern "C" JNIEXPORT jint JNICALL
Java_com_mobilefinetuner_visualizer_continuity_NativeOwnedProcess_nativeSpawn(
        JNIEnv* env, jobject, jbyteArray spec_array, jbyteArray env_array) {
    try {
        Strings spec, environment;
        if (!spec.load(env, spec_array) || !environment.load(env, env_array) ||
            spec.pointers.size() < 4 || spec.pointers[2][0] != '/') {
            if (!env->ExceptionCheck()) fail(env, "invalid trainer argv/env", EINVAL);
            return -1;
        }
        struct sigaction disposition{};
        if (sigaction(SIGCHLD, nullptr, &disposition) != 0) { fail(env, "sigaction", errno); return -1; }
        if (disposition.sa_handler != SIG_DFL || (disposition.sa_flags & SA_NOCLDWAIT)) {
            fail(env, "SIGCHLD disposition incompatible with exclusive waitpid", ECHILD);
            return -1;
        }
        auto argv = spec.pointers.data();
        auto envp = environment.pointers.data();
        pid_t parent = getpid();
        sigset_t all, original;
        sigfillset(&all);
        int mask_err = pthread_sigmask(SIG_SETMASK, &all, &original);
        if (mask_err) { fail(env, "pthread_sigmask", mask_err); return -1; }
        pid_t pid = fork();
        if (pid == 0) exec_child(parent, argv, envp);
        int err = errno;
        pthread_sigmask(SIG_SETMASK, &original, nullptr);
        if (pid < 0) fail(env, "fork trainer", err);
        return pid;
    } catch (const std::bad_alloc&) {
        fail(env, "allocate trainer argv/env", ENOMEM);
        return -1;
    }
}

// Only called under this child's Kotlin monitor. Never signal after a successful reap
// or ECHILD. An exit between WNOHANG and kill leaves an unreaped zombie pinning the PID.
extern "C" JNIEXPORT jint JNICALL
Java_com_mobilefinetuner_visualizer_continuity_NativeOwnedProcess_nativeCollect(
        JNIEnv* env, jobject, jint pid, jboolean terminate) {
    if (pid <= 1) return -2;
    int status = 0;
    auto result = wait_child(pid, &status, WNOHANG);
    if (result == pid && exit_code(status) >= 0) return exit_code(status);
    if (result >= 0 && !terminate) return -1;
    if (result >= 0) {
        if (kill(pid, SIGKILL) < 0 && errno != ESRCH) { fail(env, "kill trainer", errno); return -1; }
        do { result = wait_child(pid, &status, 0); } while (result == pid && exit_code(status) < 0);
        if (result == pid) return exit_code(status);
    }
    if (errno == ECHILD) return -2;
    fail(env, "waitpid trainer", errno);
    return -1;
}
