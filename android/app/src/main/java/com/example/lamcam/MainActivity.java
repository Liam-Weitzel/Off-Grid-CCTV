package com.example.lamcam;

import androidx.appcompat.app.AppCompatActivity;

import android.net.Uri;
import android.os.Bundle;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        WebView wv = (WebView) findViewById(R.id.webView);
        wv.setWebViewClient(new WebViewClient());
        String URL = "http://192.168.43.158:8080/viewCamera?url=ws://192.168.43.158:8081/";
        wv.loadUrl(String.valueOf(Uri.parse(URL)));
        WebSettings webSettings = wv.getSettings();
        webSettings.setJavaScriptEnabled(true);
    }
}