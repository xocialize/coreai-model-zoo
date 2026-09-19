// Copyright 2026.
// ⚠️ DRAFT (compile/run on macOS 27). Minimal CLI to validate the N-state Core AI runner on a
// Mac with a fast loop, BEFORE the iOS app. Feeds raw token ids (no tokenizer yet) so it isolates
// the engine: load a stateful .aimodel, prefill the prompt ids, greedy-decode N tokens, print ids.
// Verify the first generated id against the conversion oracle (e.g. _smoke/qwen3_5_ref.pt decode_token).
//
//   (release Xcode 27 (macOS 27 SDK) is the active toolchain — no DEVELOPER_DIR override needed (AB-A-0077, 2026-09-17))
//   swift run coreai-run --model <bundle.aimodel> --vocab 151936 --prompt "2,1037,4521,9,108,2516,65190,17" --max 5
//
// (Get vocab/prompt-ids from the conversion oracle. Tokenizer-driven text I/O lives in the iOS app.)

import CoreAI
import CoreAIRunner
import Foundation

func arg(_ name: String, default def: String? = nil) -> String? {
    let a = CommandLine.arguments
    if let i = a.firstIndex(of: name), i + 1 < a.count { return a[i + 1] }
    return def
}

guard let modelPath = arg("--model") else {
    print("usage: coreai-run --model <bundle.aimodel> --vocab <N> --prompt \"id,id,...\" [--max 5] [--ctx 4096]")
    exit(2)
}
let vocab = Int(arg("--vocab", default: "0")!) ?? 0
let ctx = Int(arg("--ctx", default: "4096")!) ?? 4096
let maxNew = Int(arg("--max", default: "5")!) ?? 5
let promptIds = (arg("--prompt", default: "")!)
    .split(separator: ",").compactMap { Int32($0.trimmingCharacters(in: .whitespaces)) }

guard vocab > 0, !promptIds.isEmpty else {
    print("error: --vocab and --prompt are required (--prompt is comma-separated token ids)")
    exit(2)
}

let url = URL(fileURLWithPath: modelPath)

let sema = DispatchSemaphore(value: 0)
Task {
    do {
        var spec = HybridCoreAIEngine.Spec(vocabSize: vocab)
        spec.maxContextLength = ctx
        let engine = try await HybridCoreAIEngine(modelURL: url, spec: spec)
        let t0 = Date()
        let out = try await engine.generateGreedy(promptTokens: promptIds, maxNewTokens: maxNew)
        let dt = Date().timeIntervalSince(t0)
        print("generated ids: \(out)")
        print(String(format: "%.1f tok/s (%d new tokens in %.2fs)", Double(maxNew) / dt, maxNew, dt))
    } catch {
        print("error: \(error)")
    }
    sema.signal()
}
sema.wait()
