build:
	docker-compose build

run:
	docker-compose up

export:
	docker cp onnx_converter_instance:/home/app/output/model_float32.tflite ./model_float32.tflite
	docker cp onnx_converter_instance:/home/app/output/model_float16.tflite ./model_float16.tflite

clean:
	docker-compose down
	docker-compose rm -f
	rm -rf output