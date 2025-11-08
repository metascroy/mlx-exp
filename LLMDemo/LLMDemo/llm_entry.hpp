// llm_entry.hpp
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

// Returns 0 on success; non-zero on failure.
// prog_json_path:   absolute path to "prog.json" (exported program)
// consts_path:      absolute path to "consts.safetensors" (model weights)
// prompt_ids_path:  absolute path to a text file with space-separated token ids
// max_new_tokens:   e.g. 64
// print_batch:      how many tokens to buffer before invoking on_tokens callback (>=1)
// on_tokens:        called on background thread with a pointer to `count` newly generated ids
// on_log:           optional logging callback (can be null)
int run_llm_generation(const char* prog_json_path,
                       const char* consts_path,
                       const char* prompt_ids_path,
                       int max_new_tokens,
                       int print_batch,
                       void (*on_tokens)(const int* ids, int count, void* user),
                       void* user,
                       void (*on_log)(const char* msg, void* user));

#ifdef __cplusplus
}
#endif
