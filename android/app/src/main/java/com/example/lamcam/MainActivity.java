package com.example.lamcam;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.Bundle;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import androidx.annotation.DrawableRes;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.content.res.AppCompatResources;

import com.mapbox.geojson.Point;
import com.mapbox.maps.CameraOptions;
import com.mapbox.maps.MapView;
import com.mapbox.maps.MapboxMap;
import com.mapbox.maps.Style;
import com.mapbox.maps.extension.style.StyleContract;
import com.mapbox.maps.extension.style.StyleExtensionImpl;
import com.mapbox.maps.extension.style.sources.generated.RasterDemSource;
import com.mapbox.maps.extension.style.terrain.generated.Terrain;
import com.mapbox.maps.plugin.Plugin;
import com.mapbox.maps.plugin.annotation.AnnotationConfig;
import com.mapbox.maps.plugin.annotation.AnnotationPlugin;
import com.mapbox.maps.plugin.annotation.AnnotationType;
import com.mapbox.maps.plugin.annotation.generated.PointAnnotationManager;
import com.mapbox.maps.plugin.annotation.generated.PointAnnotationOptions;

public class MainActivity extends AppCompatActivity {
    private MapView mapView;
    private MapboxMap mapboxMap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mapView = findViewById(R.id.mapView);
        mapboxMap = mapView.getMapboxMap();
        mapboxMap.setCamera(
                new CameraOptions.Builder()
                        .center(Point.fromLngLat(-98.0, 39.5))
                        .pitch(0.0)
                        .zoom(14.0)
                        .bearing(0.0)
                        .build()
        );
        mapboxMap.loadStyle(createStyle());

        WebView wv = (WebView) findViewById(R.id.webView);
        wv.setWebViewClient(new WebViewClient());
        String URL = "http://192.168.43.158:8080/viewCamera?url=ws://192.168.43.158:8081/";
        wv.loadUrl(String.valueOf(Uri.parse(URL)));
        WebSettings webSettings = wv.getSettings();
        webSettings.setJavaScriptEnabled(true);
    }

    private StyleContract.StyleExtension createStyle() {
        StyleExtensionImpl.Builder builder = new StyleExtensionImpl.Builder(Style.SATELLITE);

        RasterDemSource rasterDemSource = new RasterDemSource(new RasterDemSource.Builder("TERRAIN_SOURCE").tileSize(514));
        rasterDemSource.url("mapbox://mapbox.mapbox-terrain-dem-v1");
        builder.addSource(rasterDemSource);

        Terrain terrain = new Terrain("TERRAIN_SOURCE");
        terrain.exaggeration(1.5);
        builder.setTerrain(terrain);

        return builder.build();
    }

    private void addAnnotationToMap() {
        Bitmap bitmap = bitmapFromDrawableRes(this, R.drawable.red_marker);
        if (bitmap != null) {
            AnnotationPlugin annotationApi = mapView.getPlugin(Plugin.MAPBOX_ANNOTATION_PLUGIN_ID);
            PointAnnotationManager pointAnnotationManager = (PointAnnotationManager) annotationApi.createAnnotationManager(AnnotationType.PointAnnotation, new AnnotationConfig());
            PointAnnotationOptions pointAnnotationOptions = new PointAnnotationOptions()
                    .withPoint(Point.fromLngLat(18.06, 59.31))
                    .withIconImage(bitmap);
            pointAnnotationManager.create(pointAnnotationOptions);
        }
    }

    private Bitmap bitmapFromDrawableRes(Context context, @DrawableRes int resourceId) {
        return convertDrawableToBitmap(AppCompatResources.getDrawable(context, resourceId));
    }

    private Bitmap convertDrawableToBitmap(Drawable sourceDrawable) {
        if (sourceDrawable == null) {
            return null;
        }
        if (sourceDrawable instanceof BitmapDrawable) {
            return ((BitmapDrawable) sourceDrawable).getBitmap();
        } else {
            Drawable drawable = sourceDrawable.getConstantState().newDrawable().mutate();
            Bitmap bitmap = Bitmap.createBitmap(
                    drawable.getIntrinsicWidth(), drawable.getIntrinsicHeight(),
                    Bitmap.Config.ARGB_8888
            );
            Canvas canvas = new Canvas(bitmap);
            drawable.setBounds(0, 0, canvas.getWidth(), canvas.getHeight());
            drawable.draw(canvas);
            return bitmap;
        }
    }

}