// Minimal stand-in for gufo's cli/serve logging for out-of-tree builds.
//
// The vendored deepseek_v4_flash engine logs a handful of informational
// lines through gufo::server::Logger, which lives in the CLI layer and is
// not exported to extensions. Match the subset of the API the engine uses
// (LogFormatted/kInfo/kWarn) and write to stderr with the same line shape:
//   YYYY-MM-DD HH:MM:SS [LEVEL] [component] message
#ifndef GUFO_DS41_LOGGING_SHIM_HPP_
#define GUFO_DS41_LOGGING_SHIM_HPP_

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <string_view>

namespace gufo::server {

enum class LogLevel : std::uint8_t { kDebug, kInfo, kWarn, kError };

class Logger {
public:
  static void SetLevel(LogLevel level) { level_ = level; }
  static LogLevel Level() { return level_; }
  static bool Enabled(LogLevel level) { return level >= level_; }

  static void Log(LogLevel level, std::string_view component,
                  std::string_view message) {
    if (!Enabled(level)) return;
    std::fprintf(stderr, "%s [%s] [%.*s] %.*s\n", Stamp(), Name(level),
                 static_cast<int>(component.size()), component.data(),
                 static_cast<int>(message.size()), message.data());
  }

  static void LogFormatted(LogLevel level, std::string_view component,
                           const char* format, ...) {
    if (!Enabled(level)) return;
    char buf[2048];
    va_list ap;
    va_start(ap, format);
    std::vsnprintf(buf, sizeof(buf), format, ap);
    va_end(ap);
    Log(level, component, buf);
  }

private:
  static const char* Name(LogLevel level) {
    switch (level) {
      case LogLevel::kDebug: return "DEBUG";
      case LogLevel::kWarn:  return "WARN";
      case LogLevel::kError: return "ERROR";
      default:               return "INFO";
    }
  }
  static const char* Stamp() {
    static thread_local char stamp[32];
    std::time_t now = std::time(nullptr);
    std::tm tm {};
    localtime_r(&now, &tm);
    std::strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", &tm);
    return stamp;
  }
  inline static LogLevel level_ = LogLevel::kInfo;
};

}  // namespace gufo::server

#endif  // GUFO_DS41_LOGGING_SHIM_HPP_
