package com.example.lamcam;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.DrawableRes;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.content.res.AppCompatResources;

import com.mapbox.geojson.Point;
import com.mapbox.maps.AnnotatedFeature;
import com.mapbox.maps.CameraBoundsOptions;
import com.mapbox.maps.CameraOptions;
import com.mapbox.maps.CoordinateBounds;
import com.mapbox.maps.MapView;
import com.mapbox.maps.MapboxMap;
import com.mapbox.maps.Style;
import com.mapbox.maps.ViewAnnotationOptions;
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
import com.mapbox.maps.viewannotation.ViewAnnotationManager;

public class view_map extends AppCompatActivity {
    private MapView mapView;
    private MapboxMap mapboxMap;
    private String serverIp;

    private static final int MAX_CLICK_DURATION = 1000;
    private static final int MAX_CLICK_DISTANCE = 15;
    private long pressStartTime;
    private float pressedX;
    private float pressedY;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_view_map);
        Intent intent = getIntent();
        serverIp = intent.getStringExtra("serverIp");

        mapView = findViewById(R.id.mapView);
        mapboxMap = mapView.getMapboxMap();

        mapboxMap.setCamera(
                new CameraOptions.Builder()
                        .center(Point.fromLngLat(15.013785520105046, 36.90453150945084))
                        .pitch(0.0)
                        .zoom(16.0)
                        .bearing(-50.0)
                        .build()
        );

        CameraBoundsOptions options = new CameraBoundsOptions.Builder()
                .bounds(CoordinateBounds.singleton(Point.fromLngLat(15.013785520105046, 36.90453150945084)))
                .maxPitch(60.0)
                .minZoom(15.0)
                .maxZoom(17.0)
                .build();
        mapboxMap.setBounds(options);
        mapboxMap.loadStyle(createStyle());

        addAnnotationToMap();
        clickHandler();
    }

    private StyleContract.StyleExtension createStyle() {
        StyleExtensionImpl.Builder builder = new StyleExtensionImpl.Builder(Style.SATELLITE);

        RasterDemSource rasterDemSource = new RasterDemSource(new RasterDemSource.Builder("TERRAIN_SOURCE").tileSize(514));
        rasterDemSource.url("mapbox://mapbox.mapbox-terrain-dem-v1");
        builder.addSource(rasterDemSource);

        Terrain terrain = new Terrain("TERRAIN_SOURCE");
        terrain.exaggeration(2.5);
        builder.setTerrain(terrain);

        return builder.build();
    }

    private void addAnnotationToMap() {
        Bitmap bitmap = bitmapFromDrawableRes(this, R.drawable.camera);
        if (bitmap != null) {
            AnnotationPlugin annotationApi = mapView.getPlugin(Plugin.MAPBOX_ANNOTATION_PLUGIN_ID);
            PointAnnotationManager pointAnnotationManager = (PointAnnotationManager) annotationApi.createAnnotationManager(AnnotationType.PointAnnotation, new AnnotationConfig());
            PointAnnotationOptions pointAnnotationOptions = new PointAnnotationOptions()
                    .withPoint(Point.fromLngLat(15.013785520105046, 36.90453150945084))
                    .withIconImage(bitmap);
            pointAnnotationManager.create(pointAnnotationOptions);

            pointAnnotationManager.addClickListener(pointAnnotation -> {
                openCameraPreview();
                return false;
            });
        }
    }

    private void openCameraPreview() {
        camera_preview cameraPreview = new camera_preview(this);
        cameraPreview.setLayoutParams(new ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)); // Set the desired layout parameters
        cameraPreview.init(serverIp, "8080", "8084");
        AnnotatedFeature annotatedFeature = new AnnotatedFeature(Point.fromLngLat(15.013785520105046, 36.90453150945084));
        ViewAnnotationOptions options = new ViewAnnotationOptions.Builder()
                .annotatedFeature(annotatedFeature)
                .width(800.0)
                .height(800.0)
                .allowOverlap(true)
                .visible(true)
                .build();

        ViewAnnotationManager viewAnnotationManager = mapView.getViewAnnotationManager();
        viewAnnotationManager.addViewAnnotation(cameraPreview, options);
    }

    private void clickHandler() {
        mapView.setOnTouchListener(new View.OnTouchListener() {
            @SuppressLint("ClickableViewAccessibility")
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                switch (event.getActionMasked()) {
                    case MotionEvent.ACTION_DOWN: {
                        pressStartTime = System.currentTimeMillis();
                        pressedX = event.getX();
                        pressedY = event.getY();
                        break;
                    }
                    case MotionEvent.ACTION_UP:
                        long pressDuration = System.currentTimeMillis() - pressStartTime;
                        if (pressDuration < MAX_CLICK_DURATION && distance(pressedX, pressedY, event.getX(), event.getY()) < MAX_CLICK_DISTANCE) {
                            ViewAnnotationManager viewAnnotationManager = mapView.getViewAnnotationManager();
                            if(!viewAnnotationManager.getAnnotations().isEmpty()) {
                                viewAnnotationManager.removeAllViewAnnotations();
                            }
                        }
                        break;
                    default:
                        break;
                }
                return false;
            }
        });
    }

    private float distance(float x1, float y1, float x2, float y2) {
        float dx = x1 - x2;
        float dy = y1 - y2;
        float distanceInPx = (float) Math.sqrt(dx * dx + dy * dy);
        return pxToDp(distanceInPx);
    }

    private float pxToDp(float px) {
        return px / getResources().getDisplayMetrics().density;
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