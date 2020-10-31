#|/bin/bash

CONTAINERS=0
declare MODEL_NAME=()

while [ "$1" ]; do
    case "$1" in
        --model_name)
            for name in "${@: 2}"; do
                if [ $name != "-c" ]; then
                    echo $name
                    MODEL_NAME+=($name)
                fi
            done
            declare NUM_MODELS="${#MODEL_NAME[@]}"
            shift "$(($NUM_MODELS+1))"
            ;;
        -c)
            CONTAINERS=1
            shift
            ;;
        --)
            shift
            break;
    esac
done

echo "CONTAINERS=$CONTAINERS"
echo "MODEL_NAME=${MODEL_NAME[*]}"
echo "NUM_MODELS=$NUM_MODELS"

# TODO: Check that model names exists as folders
FOLDERS_IN_STRING=$(ls -l models | grep '^d' | sed -e 's/.* //' | paste -s -d':' -)
echo "FOLDERS_IN_STRING=$FOLDERS_IN_STRING"
IFS=':' read -ra FOLDERS_IN_MODELS <<< "$FOLDERS_IN_STRING"
echo "${#FOLDERS_IN_MODELS[@]}"

INTERSECTIONS=()

for m in "${MODEL_NAME[@]}"; do
    for f in "${FOLDERS_IN_MODELS[@]}"; do
        if [[ "$m" == "$f" ]]; then
            INTERSECTIONS+=( "$m" )
        fi
    done
done

if [ "${#INTERSECTIONS[@]}" -lt 1 ]; then
    echo "Folders with names: ${MODEL_NAME[*]} not found."
    exit 1
fi

if [ $NUM_MODELS -gt 1 ] && [ $CONTAINERS == 0 ];then
    echo "Option 1"
    add_config(){
        echo "config {
                name: '$1'
                base_path: '/models/$1/'
                model_platform: 'tensorflow'
                }"
    }

    MODEL_CONFIG="model_config_list{"
    for model in "${MODEL_NAME[@]}"; do
        echo $model
        MODEL_CONFIG="$MODEL_CONFIG\n\t$(add_config $model)"
    done
    MODEL_CONFIG="$MODEL_CONFIG\n}"

    echo "$MODEL_CONFIG" >> "models/config/models.config"
    docker run -t --rm -p 8501:8501 -v "$(pwd)/models:/models" tensorflow/serving \
    --model_config_file=/models/config/models.config \
    --allow_version_labels_for_unavailable_models=true
fi

if [ $NUM_MODELS -gt 1 ] && [ $CONTAINERS == 1 ];then
    echo "option 2"
    COUNT=0
    for model in "${MODEL_NAME[@]}"; do
        PORT_CONTAINER=$((8501 + $COUNT))
        docker run -t --rm -p "$PORT_CONTAINER:$PORT_CONTAINER" -v "$(pwd)/models:/models" \
        -e MODEL_NAME="$model" tensorflow/serving &
        echo "Model $model served in port $PORT_CONTAINER."
        COUNT+=1
    done
fi

if [ $NUM_MODELS -eq 1 ]; then
    echo "option 3"
    docker run -t --rm -p 8501:8501 \
    -v "$(pwd)/models:/models" -e MODEL_NAME="${MODEL_NAME[0]}" \
    tensorflow/serving
fi

