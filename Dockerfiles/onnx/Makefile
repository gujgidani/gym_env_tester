build:
	docker-compose build

run:
	docker-compose up

export:
	docker save minimal_python-plc_nn_converter_instance > plc_nn_converter_instance.tar

clean:
	docker-compose down
	docker-compose rm -f
	rm -rf output