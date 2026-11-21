package com.mobilefinetuner.visualizer.continuity;

import android.app.Activity;
import android.app.ActivityManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Debug;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.os.Messenger;
import android.os.Process;
import android.os.RemoteException;
import android.view.WindowManager;
import android.widget.TextView;

import java.util.ArrayList;

// Framework-only: this component runs in the standalone test APK, outside instrumentation.
public final class PressureActivity extends Activity {
    static final int READY = 1, GROW = 2, RELEASE = 3, REFUSED = 4;
    static final int CHUNK_MB = 32, CAP_MB = 512;
    private final ArrayList<Bitmap> pages = new ArrayList<>();
    private final Handler timer = new Handler(Looper.getMainLooper());
    private Messenger client;
    private boolean ready;
    private final Messenger commands = new Messenger(new Handler(Looper.getMainLooper(), message -> {
        if (isFinishing()) return true;
        if (message.what == RELEASE) {
            release();
            report(RELEASE, 0);
            finish();
        } else if (message.what == GROW) {
            ActivityManager.MemoryInfo mem = new ActivityManager.MemoryInfo();
            getSystemService(ActivityManager.class).getMemoryInfo(mem);
            if (pages.size() * CHUNK_MB >= CAP_MB || mem.lowMemory ||
                    mem.availMem < Math.max(2048L << 20, mem.threshold + (512L << 20))) {
                report(REFUSED, pages.size() * CHUNK_MB);
            } else {
                // Software pixels live in native memory on the supported API levels.
                Bitmap page = Bitmap.createBitmap(2048, 4096, Bitmap.Config.ARGB_8888);
                pages.add(page);
                page.eraseColor(0xff102030 + pages.size());
                report(GROW, pages.size() * CHUNK_MB);
            }
        }
        return true;
    }));

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        client = getIntent().getParcelableExtra("client");
        if (client == null || getIntent().getComponent() == null) { finish(); return; }
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        TextView text = new TextView(this);
        text.setText("Continuity test: bounded memory pressure");
        setContentView(text);
        timer.postDelayed(this::finish, 60000); // Also release if instrumentation disappears.
    }

    @Override protected void onResume() {
        super.onResume();
        if (!ready && !isFinishing()) { ready = true; report(READY, Process.myPid()); }
    }

    private void report(int event, int value) {
        Debug.MemoryInfo mem = new Debug.MemoryInfo();
        Debug.getMemoryInfo(mem);
        Message reply = Message.obtain(null, event, value, mem.getTotalPrivateDirty());
        reply.replyTo = commands;
        try { client.send(reply); } catch (RemoteException gone) { finish(); }
    }

    private void release() {
        for (Bitmap page : pages) page.recycle();
        pages.clear();
    }

    @Override protected void onDestroy() {
        timer.removeCallbacksAndMessages(null);
        release();
        super.onDestroy();
    }
}
