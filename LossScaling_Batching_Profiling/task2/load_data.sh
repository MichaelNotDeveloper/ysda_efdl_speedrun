curl -L -o data.zip "https://www.dropbox.com/scl/fi/e6oqpx6iuos7kn9m139z7/wikitext-103-raw-v1.zip?dl=0&e=1&file_subpath=%2Fwikitext-103-raw-v1&rlkey=81evwbaqfkxtckj8zhks7yied&st=6ept2pdm"
unzip data.zip -d ./data
rm -fr data.zip
rm -fr data/__MACOSX