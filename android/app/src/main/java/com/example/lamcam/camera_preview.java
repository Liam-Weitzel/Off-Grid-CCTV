package com.example.lamcam;

import android.content.Context;
import android.content.Intent;
import android.graphics.Color;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.FrameLayout;

/**
 * TODO: document your custom view class.
 */
public class camera_preview extends FrameLayout {

    private Context mContext;

    public camera_preview(Context context) {
        super(context);
        mContext = context;
    }

    @Override
    protected void onLayout(boolean changed, int l, int t, int r, int b) {
        super.onLayout(changed, l, t, r, b);
    }

    public void init(String serverIp, String apiPort, String wsPort) {
        WebView wv = new WebView(getContext());
        wv.setLayoutParams(new ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
        wv.setWebViewClient(new WebViewClient());
        wv.setBackgroundColor(Color.TRANSPARENT);
        wv.setClickable(false);
        String URL = "http://" + serverIp + ":" + apiPort + "/viewCamera?url=ws://" + serverIp + ":" + wsPort + "/";
        wv.loadUrl(URL);
        WebSettings webSettings = wv.getSettings();
        webSettings.setJavaScriptEnabled(true);
        addView(wv);

        wv.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                if (event.getAction() == MotionEvent.ACTION_UP) {
                    Intent intent = new Intent(mContext, view_camera.class);
                    intent.putExtra("wsPort", "8084");
                    intent.putExtra("serverIp", serverIp);
                    intent.putExtra("apiPort", "8080");
                    mContext.startActivity(intent);
                    return true;
                }
                return false;
            }
        });
    }
}