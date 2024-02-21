#!/usr/bin/env bash
#builds all docker images in this mono-repo and runs them using docker-compose

declare -a imageName=()
declare -a imageVersion=()
size=0

echo "The following commands will be executed:"

for dir in */
do
    currentDir=$(echo "${dir}" | sed 's/\///g')
    imageName[${size}]=${currentDir}
    #echo "${currentDir}"
    if find "${currentDir}" -type f -name "Dockerfile" | grep -q .; then

	#fetch the latest version number of each image
	fullVersion=$(sudo docker images | grep -o -P "${currentDir}.{0,20}" | grep -o -P ' v.{0,5}' | sed 's/ //g' | sed 's/v//g' | sort -V | tail -n 1)
	majorVersion=$(echo "${fullVersion}" | grep -o -P '.{0,10}\.' | sed 's/\.//g')
	minorVersion=$(echo "${fullVersion}" | grep -o -P '\..{0,10}' | sed 's/\.//g')

	if [ -z "$majorVersion" ]; then
	    majorVersion=0
	fi
	if [ -z "$minorVersion" ]; then
	    minorVersion=0	
	fi

	#echo "${currentDir}: ${fullVersion}  -  v${majorVersion}.${minorVersion}"

	#increment version no.
	if [ $1 = 'major' ]; then
	    ((majorVersion++))
	elif [ $1 = 'minor' ]; then
	    ((minorVersion++))
	fi
	newVersion=$(echo "v${majorVersion}.${minorVersion}")

	imageVersion[${size}]=${newVersion}
	#echo "${newVersion}"
	
	echo "cd ${currentDir}/ && sudo docker build . -t ${currentDir}:${newVersion} -t ${currentDir}:latest && cd ../"

	((size++))
    fi
done

read -t 1 -n 1
echo "Press any key to continue"
while [ true ] ; do
read -t 3 -n 1
if [ $? = 0 ] ; then
break ;
else
continue ;
fi
done

for (( i = 0; i <= $size; i++ ))
do
    #echo "${imageName[$i]}  ${imageVersion[$i]}"
    cd ${imageName[$i]}/ && sudo docker build . -t ${imageName[$i]}:${imageVersion[$i]} -t ${imageName[$i]}:latest && cd ../
done

sudo docker-compose up -d #runs docker compose headless mode

#sudo docker exec -it off-grid-cctv_backend_1 /bin/bash #this will ssh into the container
