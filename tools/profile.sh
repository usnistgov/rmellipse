mkdir code-profiles
filename=$(basename "$1")
output="code-profiles/${filename}.dat"
echo "profiling : ${1}"
echo "outputting to : ${output}"
uv run python -m cProfile -o "${output}" $1
uv run snakeviz "${output}"
