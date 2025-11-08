import SwiftUI

final class GenVM: ObservableObject {
  @Published var tokens: [Int] = []
  @Published var logs: [String] = []
  private let runner = LLMRunner()

  func start() {
    // look for the three files we now need
    guard let progURL = Bundle.main.url(forResource: "prog", withExtension: "json"),
          let constsURL = Bundle.main.url(forResource: "consts", withExtension: "safetensors"),
          let promptURL = Bundle.main.url(forResource: "prompt_ids", withExtension: "txt") else {
      logs.append("Missing prog.json, consts.safetensors, or prompt_ids.txt in bundle")
      return
    }

    tokens.removeAll()
    logs.removeAll()

      runner.run(
        withProgramPath: progURL.path,
        constsPath: constsURL.path,
        promptPath: promptURL.path,
        maxNewTokens: 128,
        printBatch: 1,
        tokens: { [weak self] chunk in
          DispatchQueue.main.async {
            self?.tokens.append(contentsOf: chunk.map { $0.intValue })
          }
        },
        log: { [weak self] s in
          DispatchQueue.main.async {
            self?.logs.append(s)
          }
        },
        completion: { status in
          print("Done with status \(status)")
        }
      )

  }
}

struct ContentView: View {
  @StateObject var vm = GenVM()
  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      Button("Run LLM") { vm.start() }

      Text("Tokens:")
        .font(.headline)

      ScrollView {
        Text(vm.tokens.map(String.init).joined(separator: " "))
          .font(.system(.footnote, design: .monospaced))
          .frame(maxWidth: .infinity, alignment: .leading)
      }
      .frame(maxHeight: 180)

      Text("Logs:")
        .font(.headline)

      ScrollView {
        VStack(alignment: .leading, spacing: 6) {
          ForEach(vm.logs, id: \.self) {
            Text($0).font(.caption.monospaced())
          }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
      }

      Spacer()
    }
    .padding()
  }
}
