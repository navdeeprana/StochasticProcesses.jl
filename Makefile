# Convert all notebooks to julia
ipynb_files := $(wildcard *.ipynb)
scripts: $(ipynb_files)
	@for f in $(ipynb_files); do echo "Converting $$f"; jupytext --to jl $$f; done

# Format all notebooks
format: $(ipynb_files)
	@for f in $(ipynb_files); do echo "Formatting $$f"; \
		jupytext --pipe "julia -e 'using JuliaFormatter; format_file(ARGS[1]; margin=150)' {}" $$f; done

# Convert all julia files to notebooks for working
julia_git := $(wildcard *.jl)

notebooks: $(julia_git)
	@for f in $(julia_git); do echo "Converting $$f to notebook"; jupytext --to ipynb $$f; done
