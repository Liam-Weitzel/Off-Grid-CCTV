package com.example.lamcam;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.View;
import android.widget.CheckBox;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.textfield.TextInputLayout;

public class MainActivity extends AppCompatActivity {
    private TextInputLayout serverIp;
    private CheckBox rememberIp;
    boolean isRemember;
    public static Context context;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        serverIp = findViewById(R.id.serverIpInput);
        SharedPreferences pref = getSharedPreferences("config", MODE_PRIVATE);
        isRemember = pref.getBoolean("remember",false);
        rememberIp = findViewById(R.id.rememberIp);
        rememberIp.setChecked(isRemember);

        if(isRemember) {
            serverIp.getEditText().setText(pref.getString("serverIp",""));
        }

        context = getApplicationContext();
    }

    public void connectButtonClickHandler(View view) {
        SharedPreferences pref = getSharedPreferences("config", MODE_PRIVATE);
        SharedPreferences.Editor editor = pref.edit();
        Intent intent = new Intent(this, MapActivity.class);
        intent.putExtra("serverIp", serverIp.getEditText().getText().toString());
        editor.putBoolean("remember", rememberIp.isChecked());
        if(rememberIp.isChecked()) editor.putString("serverIp", serverIp.getEditText().getText().toString());
        editor.commit();
        startActivity(intent);
    }
}