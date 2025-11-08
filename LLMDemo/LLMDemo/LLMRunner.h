// LLMRunner.h
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef void (^LLMTokensBlock)(NSArray<NSNumber*> *tokens);
typedef void (^LLMLogBlock)(NSString *message);
typedef void (^LLMCompletion)(int status);

@interface LLMRunner : NSObject

// progPath:   path to prog.json
// constsPath: path to consts.safetensors
// promptPath: path to prompt_ids.txt
- (void)runWithProgramPath:(NSString *)progPath
                 constsPath:(NSString *)constsPath
                 promptPath:(NSString *)promptPath
              maxNewTokens:(NSInteger)maxNew
                printBatch:(NSInteger)printBatch
                    tokens:(LLMTokensBlock)onTokens
                       log:(LLMLogBlock)onLog
                 completion:(LLMCompletion)onDone;

@end

NS_ASSUME_NONNULL_END
