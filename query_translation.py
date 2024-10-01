
### Decomposition
# Chain
generate_queries_decomposition = ( prompt_decomposition | model | StrOutputParser() | (lambda x: x.split("\n")))

# Run
decomposition_questions = generate_queries_decomposition.invoke({"question":question})

