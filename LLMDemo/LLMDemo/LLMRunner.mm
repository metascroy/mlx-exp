// LLMRunner.mm
#import "LLMRunner.h"
#import "llm_entry.hpp"
#import <objc/runtime.h>

@interface LLMCallbackBox : NSObject
@property (nonatomic, copy) LLMTokensBlock tokensBlock;
@property (nonatomic, copy) LLMLogBlock    logBlock;
@end
@implementation LLMCallbackBox @end

@implementation LLMRunner

// C → ObjC tokens callback
static void tokens_c_cb(const int* ids, int count, void* user) {
  id obj = (__bridge id)user;
  if (![obj isKindOfClass:[LLMCallbackBox class]]) { return; }
  LLMCallbackBox *box = (LLMCallbackBox *)obj;
  if (!box.tokensBlock) return;

  NSMutableArray<NSNumber*> *arr = [NSMutableArray arrayWithCapacity:count];
  for (int i = 0; i < count; ++i) {
    [arr addObject:@(ids[i])];
  }
  box.tokensBlock(arr);
}

// C → ObjC log callback
static void log_c_cb(const char* msg, void* user) {
  id obj = (__bridge id)user;
  if (![obj isKindOfClass:[LLMCallbackBox class]]) { return; }
  LLMCallbackBox *box = (LLMCallbackBox *)obj;
  if (!box.logBlock) return;

  box.logBlock([NSString stringWithUTF8String:msg ?: ""]);
}

- (void)runWithProgramPath:(NSString *)progPath
                 constsPath:(NSString *)constsPath
                 promptPath:(NSString *)promptPath
              maxNewTokens:(NSInteger)maxNew
                printBatch:(NSInteger)printBatch
                    tokens:(LLMTokensBlock)onTokens
                       log:(LLMLogBlock)onLog
                 completion:(LLMCompletion)onDone
{
  // Hold the Swift/ObjC blocks alive across the C call
  LLMCallbackBox *box = [LLMCallbackBox new];
  box.tokensBlock = onTokens;
  box.logBlock    = onLog;

  // Retain across the async boundary
  void *userPtr = (__bridge_retained void *)box;

  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
    int status = run_llm_generation(
      progPath.fileSystemRepresentation,    // prog.json
      constsPath.fileSystemRepresentation,  // consts.safetensors
      promptPath.fileSystemRepresentation,  // prompt_ids.txt
      (int)maxNew,
      (int)printBatch,
      onTokens ? tokens_c_cb : nullptr,
      userPtr,
      onLog ? log_c_cb : nullptr
    );

    // Release on main; then call completion
    dispatch_async(dispatch_get_main_queue(), ^{
      CFRelease(userPtr);
      if (onDone) onDone(status);
    });
  });
}

@end
