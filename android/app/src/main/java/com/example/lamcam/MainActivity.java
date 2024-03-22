package com.example.lamcam;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.textfield.TextInputLayout;

public class MainActivity extends AppCompatActivity {
    private TextInputLayout serverIp;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        serverIp = findViewById(R.id.serverIpInput);
    }

    public void connectButtonClickHandler(View view) {
        Intent intent = new Intent(this, view_map.class);
        intent.putExtra("serverIp", serverIp.getEditText().getText().toString());
        startActivity(intent);
    }
}