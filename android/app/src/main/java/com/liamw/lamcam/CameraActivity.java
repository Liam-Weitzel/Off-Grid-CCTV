package com.liamw.lamcam;

import android.content.Intent;
import android.os.Bundle;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import androidx.appcompat.app.AppCompatActivity;

public class CameraActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);
        Intent intent = getIntent();

        WebView wv = (WebView) findViewById(R.id.webView);
        wv.setWebViewClient(new WebViewClient());
        String URL = "http://" + intent.getStringExtra("serverIp") + ":" + intent.getStringExtra("apiPort") + "/viewCamera?url=ws://" + intent.getStringExtra("serverIp") + ":" + intent.getStringExtra("wsPort") + "/";;
        wv.loadUrl(URL);
        WebSettings webSettings = wv.getSettings();
        webSettings.setJavaScriptEnabled(true);
    }
}