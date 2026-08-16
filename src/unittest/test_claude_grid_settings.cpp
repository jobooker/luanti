// Luanti
// SPDX-License-Identifier: LGPL-2.1-or-later
//
// claude_grid: the "volume" -> "grid" rename of 2026-08-16 renamed three
// SETTINGS, which is the dangerous half of a rename. A conf or a leftover
// claude_settings_patch.conf still saying `claude_volume_follow = 0` must
// not be silently ignored -- that exact failure cost a session on
// 2026-08-16 (the conf on disk said 1, the client ran 0, and grepping the
// conf was a blind instrument). These tests prove the old name is LOUD.
//
// See spec/environment-laws.md "The volume -> grid rename (2026-08-16)".

#include "test.h"

#include "client/game.h"
#include "log_internal.h"
#include "settings.h"

class TestClaudeGridSettings : public TestBase
{
public:
	TestClaudeGridSettings() { TestManager::registerTestModule(this); }
	const char *getName() { return "TestClaudeGridSettings"; }

	void runTests(IGameDef *gamedef);

	void testLegacyNameIsLoud();
	void testNewNameIsSilent();
	void testWarnsOncePerName();
};

static TestClaudeGridSettings g_test_instance;

void TestClaudeGridSettings::runTests(IGameDef *gamedef)
{
	TEST(testLegacyNameIsLoud);
	TEST(testNewNameIsSilent);
	TEST(testWarnsOncePerName);
}

////////////////////////////////////////////////////////////////////////////////

// Every legacy name is found, and each one names its replacement in a
// line an operator can actually see.
void TestClaudeGridSettings::testLegacyNameIsLoud()
{
	Settings conf;
	conf.set("claude_volume_follow", "0");
	conf.set("claude_volume_debug", "3");
	conf.set("claude_grid_debug", "3"); // the new name alongside: not a hit
	conf.set("video_driver", "opengl3");

	CaptureLogOutput capture(g_logger);
	int found = claudeWarnRenamedSettings(&conf, "unittest");
	auto logs = capture.take();

	UASSERTEQ(int, found, 2);

	std::string all;
	for (const auto &e : logs)
		all += e.text + "\n";
	// The old name, the new name, and the fact it does nothing.
	UASSERT(all.find("claude_volume_follow") != std::string::npos);
	UASSERT(all.find("claude_grid_follow") != std::string::npos);
	UASSERT(all.find("claude_volume_debug") != std::string::npos);
	UASSERT(all.find("claude_grid_debug") != std::string::npos);
	UASSERT(all.find("unittest") != std::string::npos);
	// LOUD: at least one line at action level or above.
	bool loud = false;
	for (const auto &e : logs)
		loud = loud || e.level <= LL_ACTION;
	UASSERT(loud);
}

// A conf that has already been migrated says nothing at all. A check that
// cries wolf on a correct conf gets ignored, which is the same as not
// having one.
void TestClaudeGridSettings::testNewNameIsSilent()
{
	Settings conf;
	conf.set("claude_grid_follow", "1");
	conf.set("claude_grid_debug", "3");
	conf.set("claude_grid_snapshot", "tour_1");

	CaptureLogOutput capture(g_logger);
	int found = claudeWarnRenamedSettings(&conf, "unittest");
	auto logs = capture.take();

	UASSERTEQ(int, found, 0);
	for (const auto &e : logs)
		UASSERT(e.text.find("IGNORED SETTING") == std::string::npos);
}

// The patch file is re-read at ~1 Hz, so a name that warned on every poll
// would bury the log. Warn once per name; keep REPORTING it every time, so
// a caller can still act on it.
void TestClaudeGridSettings::testWarnsOncePerName()
{
	Settings conf;
	conf.set("claude_volume_snapshot", "tour_1");

	int first_found, second_found;
	size_t first_lines, second_lines;
	{
		CaptureLogOutput capture(g_logger);
		first_found = claudeWarnRenamedSettings(&conf, "unittest");
		first_lines = capture.take().size();
	}
	{
		CaptureLogOutput capture(g_logger);
		second_found = claudeWarnRenamedSettings(&conf, "unittest");
		second_lines = capture.take().size();
	}

	UASSERTEQ(int, first_found, 1);
	UASSERTEQ(int, second_found, 1); // still reported to the caller
	UASSERT(first_lines > 0);
	UASSERTEQ(size_t, second_lines, 0); // but not logged twice
}
