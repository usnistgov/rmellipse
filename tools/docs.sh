echo "arguments: serve open clean html html-multiversioned html-multiversioned-ci"
echo "You have provided the following arguments $1"
set -e
if [ $# -eq 0 ]
then
	set -e
	echo "No arguments supplied"
elif [ $1 = "serve" ]
then
	uv run -m http.server --bind 127.0.0.1 8000 -d docs/build
elif [ $1 = "open" ]
then
	explorer "http://127.0.0.1:8000"
elif [ $1 = "clean" ]
then
	set -e
	rm -rf docs/build
	rm -rf docs/source/auto_examples
elif [ $1 = "html" ]
then
	set -e
	uv run sphinx-build -b html docs/source docs/build

elif [ $1 = "html-multiversioned" ]
then
	set -e
	uv run sphinx-multiversion docs/source docs/build
	cp docs/source/reroute_to_stable.html	docs/build/index.html
elif [ $1 = "html-multiversioned-ci" ]
then
	set -e
	uv run sphinx-multiversion docs/source public
	cp docs/source/reroute_to_stable.html	public/index.html
fi
exit 0