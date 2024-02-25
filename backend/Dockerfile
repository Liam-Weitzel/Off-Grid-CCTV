FROM ubuntu:16.04

WORKDIR /server

COPY . .

ENV OPENCV4NODEJS_DISABLE_AUTOBUILD=1

RUN apt-get update && \
apt-get install -y build-essential curl && \
apt-get install -y --no-install-recommends wget unzip git python cmake && \
curl -sL https://deb.nodesource.com/setup_11.x | bash && \
apt-get install -y nodejs && node -v && npm -v && \
rm -rf /var/lib/apt/lists/* && \
mkdir opencv && \
cd opencv && \
wget https://github.com/Itseez/opencv/archive/3.4.15.zip --no-check-certificate -O opencv-3.4.15.zip && \
unzip opencv-3.4.15.zip && \
#wget https://github.com/Itseez/opencv_contrib/archive/3.4.15.zip --no-check-certificate -O opencv_contrib-3.4.15.zip; && \
#unzip opencv_contrib-3.4.15.zip; && \
mkdir opencv-3.4.15/build && \
cd opencv-3.4.15/build && \
cmake_flags="-D CMAKE_BUILD_TYPE=RELEASE \
  -D BUILD_EXAMPLES=OFF \
  -D BUILD_DOCS=OFF \
  -D BUILD_TESTS=OFF \
  -D BUILD_PERF_TESTS=OFF \
  -D BUILD_JAVA=OFF \
  -D BUILD_opencv_apps=OFF \
  -D BUILD_opencv_aruco=OFF \
  -D BUILD_opencv_bgsegm=OFF \
  -D BUILD_opencv_bioinspired=OFF \
  -D BUILD_opencv_ccalib=OFF \
  -D BUILD_opencv_datasets=OFF \
  -D BUILD_opencv_dnn_objdetect=OFF \
  -D BUILD_opencv_dpm=OFF \
  -D BUILD_opencv_fuzzy=OFF \
  -D BUILD_opencv_hfs=OFF \
  -D BUILD_opencv_java_bindings_generator=OFF \
  -D BUILD_opencv_js=OFF \
  -D BUILD_opencv_img_hash=OFF \
  -D BUILD_opencv_line_descriptor=OFF \
  -D BUILD_opencv_optflow=OFF \
  -D BUILD_opencv_phase_unwrapping=OFF \
  -D BUILD_opencv_python3=OFF \
  -D BUILD_opencv_python_bindings_generator=OFF \
  -D BUILD_opencv_reg=OFF \
  -D BUILD_opencv_rgbd=OFF \
  -D BUILD_opencv_saliency=OFF \
  -D BUILD_opencv_shape=OFF \
  -D BUILD_opencv_stereo=OFF \
  -D BUILD_opencv_stitching=OFF \
  -D BUILD_opencv_structured_light=OFF \
  -D BUILD_opencv_superres=OFF \
  -D BUILD_opencv_surface_matching=OFF \
  -D BUILD_opencv_ts=OFF \
  -D BUILD_opencv_xobjdetect=OFF \
  -D BUILD_opencv_xphoto=OFF \
  -D CMAKE_INSTALL_PREFIX=/usr/local" && \
#cmake_flags="$cmake_flags -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.15/modules"; && \
echo $cmake_flags && \
cmake $cmake_flags .. && \
make -j $(nproc) && \
make install && \
sh -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf' && \
ldconfig && \
cd ../../../ && \
rm -rf opencv && \
npm install -g opencv4nodejs --unsafe-perm && \
apt-get update && \
apt-get -y install ffmpeg && \
npm install && \
apt-get purge -y build-essential curl wget unzip git cmake && \
apt-get autoremove -y --purge

CMD ["node", "server.js"]
