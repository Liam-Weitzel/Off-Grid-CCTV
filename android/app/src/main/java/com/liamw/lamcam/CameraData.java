package com.liamw.lamcam;

public class CameraData {
        private String port;
        private String name;
        private String path;
        private String httpPort;
        private String wsPort;
        private String ffmpegPort;
        private String lat;
        private String lon;
        private String camFps;
        private String camResolution;
        private String bv;
        private String maxRate;
        private String bufSize;

        public String getPort() {
            return port;
        }

        public void setPort(String port) {
            this.port = port;
        }

        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public String getPath() {
            return path;
        }

        public void setPath(String path) {
            this.path = path;
        }

        public String getHttpPort() {
            return httpPort;
        }

        public void setHttpPort(String httpPort) {
            this.httpPort = httpPort;
        }

        public String getWsPort() {
            return wsPort;
        }

        public void setWsPort(String wsPort) {
            this.wsPort = wsPort;
        }

        public String getFfmpegPort() {
            return ffmpegPort;
        }

        public void setFfmpegPort(String ffmpegPort) {
            this.ffmpegPort = ffmpegPort;
        }

        public String getLat() {
            return lat;
        }

        public void setLat(String lat) {
            this.lat = lat;
        }

        public String getLon() {
            return lon;
        }

        public void setLon(String lon) {
            this.lon = lon;
        }

        public String getCamFps() {
            return camFps;
        }

        public void setCamFps(String camFps) {
            this.camFps = camFps;
        }

        public String getCamResolution() {
            return camResolution;
        }

        public void setCamResolution(String camResolution) {
            this.camResolution = camResolution;
        }

        public String getBv() { return bv; }

        public void setBv(String bv) {
            this.bv = bv;
        }

        public String getMaxRate() {
            return maxRate;
        }

        public void setMaxRate(String maxRate) {
            this.maxRate = maxRate;
        }

        public String getBufSize() {
            return bufSize;
        }

        public void setBufSize(String bufSize) {
            this.bufSize = bufSize;
        }
}
