package com.liamw.lamcam;

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
import android.widget.Toast;

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

import java.util.List;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;
import retrofit2.http.GET;

public class MapActivity extends AppCompatActivity {
    private MapView mapView;
    private MapboxMap mapboxMap;
    private String serverIp;
    private static final String apiPort = "8080";

    private static final int MAX_CLICK_DURATION = 1000;
    private static final int MAX_CLICK_DISTANCE = 15;
    private long pressStartTime;
    private float pressedX;
    private float pressedY;

    interface FetchConfigs{
        @GET("/fetchConfigs")
        Call<ConfigsData> getConfigs();
    }

    interface FetchCameras{
        @GET("/fetchCameras")
        Call<List<CameraData>> getCameras();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_map);
        Intent intent = getIntent();
        serverIp = intent.getStringExtra("serverIp");

        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl("http://" + serverIp + ":" + apiPort + "/")
                .addConverterFactory(GsonConverterFactory.create())
                .build();

        FetchConfigs configs = retrofit.create(FetchConfigs.class);

        configs.getConfigs().enqueue(new Callback<ConfigsData>() {
            @Override
            public void onResponse(Call<ConfigsData> call, Response<ConfigsData> response) {
                mapboxMap.setCamera(
                        new CameraOptions.Builder()
                                .center(Point.fromLngLat(Double.parseDouble(response.body().getLongitude()), Double.parseDouble(response.body().getLatitude())))
                                .pitch(Double.parseDouble(response.body().getPitch()))
                                .zoom(Double.parseDouble(response.body().getZoom()))
                                .bearing(Double.parseDouble(response.body().getBearing()))
                                .build()
                );

                CameraBoundsOptions options = new CameraBoundsOptions.Builder()
                        .bounds(CoordinateBounds.singleton(Point.fromLngLat(Double.parseDouble(response.body().getLongitude()), Double.parseDouble(response.body().getLatitude()))))
                        .maxPitch(60.0)
                        .minZoom(Double.parseDouble(response.body().getZoom())-1)
                        .maxZoom(Double.parseDouble(response.body().getZoom())+1)
                        .build();
                mapboxMap.setBounds(options);
            }

            @Override
            public void onFailure(Call<ConfigsData> call, Throwable throwable) {
                Toast.makeText(getApplicationContext(),"Cannot fetch configs from server!! Please try again",Toast.LENGTH_SHORT).show();
            }
        });

        FetchCameras cameras = retrofit.create(FetchCameras.class);

        cameras.getCameras().enqueue(new Callback<List<CameraData>>() {
            @Override
            public void onResponse(Call<List<CameraData>> call, Response<List<CameraData>> response) {
                for(CameraData camera : response.body()){
                    addAnnotationToMap(camera);
                }
            }

            @Override
            public void onFailure(Call<List<CameraData>> call, Throwable throwable) {
                Toast.makeText(getApplicationContext(),"Cannot fetch cameras from server!! Please try again",Toast.LENGTH_SHORT).show();
            }
        });

        mapView = findViewById(R.id.mapView);
        mapboxMap = mapView.getMapboxMap();
        mapboxMap.loadStyle(createStyle());
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

    private void addAnnotationToMap(CameraData camera) {
        Bitmap bitmap = bitmapFromDrawableRes(this, R.drawable.camera);
        if (bitmap != null) {
            AnnotationPlugin annotationApi = mapView.getPlugin(Plugin.MAPBOX_ANNOTATION_PLUGIN_ID);
            PointAnnotationManager pointAnnotationManager = (PointAnnotationManager) annotationApi.createAnnotationManager(AnnotationType.PointAnnotation, new AnnotationConfig());
            PointAnnotationOptions pointAnnotationOptions = new PointAnnotationOptions()
                    .withPoint(Point.fromLngLat(Double.parseDouble(camera.getLon()), Double.parseDouble(camera.getLat())))
                    .withIconImage(bitmap);
            pointAnnotationManager.create(pointAnnotationOptions);

            pointAnnotationManager.addClickListener(pointAnnotation -> {
                openCameraPreview(camera);
                return false;
            });
        }
    }

    private void openCameraPreview(CameraData camera) {
        CameraPreview cameraPreview = new CameraPreview(this);
        cameraPreview.setLayoutParams(new ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
        cameraPreview.init(serverIp, apiPort, camera.getHttpPort());
        AnnotatedFeature annotatedFeature = new AnnotatedFeature(Point.fromLngLat(Double.parseDouble(camera.getLon()), Double.parseDouble(camera.getLat())));
        ViewAnnotationOptions options = new ViewAnnotationOptions.Builder()
                .annotatedFeature(annotatedFeature)
                .width(Double.parseDouble(camera.getCamResolution()))
                .height(Double.parseDouble(camera.getCamResolution())*0.75)
                .allowOverlap(true)
                .visible(true)
                .build();

        ViewAnnotationManager viewAnnotationManager = mapView.getViewAnnotationManager();
        viewAnnotationManager.addViewAnnotation(cameraPreview, options);
    }

    private void clickHandler() {
        mapView.setOnTouchListener(new View.OnTouchListener() {
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