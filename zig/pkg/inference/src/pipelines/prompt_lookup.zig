const std = @import("std");

pub const Config = struct {
    min_ngram: usize = 4,
    max_ngram: usize = 12,
    max_draft: usize = 1,
    max_index_tokens: usize = 4096,

    pub fn validate(self: Config) !void {
        if (self.min_ngram == 0 or self.max_ngram < self.min_ngram or self.max_ngram > 32)
            return error.InvalidPromptLookupConfig;
        if (self.max_draft == 0 or self.max_draft > 16) return error.InvalidPromptLookupConfig;
        if (self.max_index_tokens == 0 or self.max_index_tokens > 4096)
            return error.InvalidPromptLookupConfig;
    }
};

pub const Counters = struct {
    lookup_attempts: u64 = 0,
    lookup_hits: u64 = 0,
    proposed_tokens: u64 = 0,
    accepted_proposal_tokens: u64 = 0,
    target_verify_passes: u64 = 0,
    target_verify_rows: u64 = 0,
    rollback_tokens: u64 = 0,
    zero_hit_fallbacks: u64 = 0,
};

pub const Proposal = struct {
    tokens: []const i64,
    ngram_len: usize,
    source_start: usize,
};

const capacity = 8192;
const Entry = struct { occupied: bool = false, hash: u64 = 0, start: usize = 0 };

pub const Proposer = struct {
    config: Config,
    table: [capacity]Entry = @splat(.{}),
    counters: Counters = .{},

    pub fn init(config: Config) !Proposer {
        try config.validate();
        return .{ .config = config };
    }

    fn tokenHash(tokens: []const i64) u64 {
        var hash: u64 = 0xcbf29ce484222325;
        for (tokens) |token| {
            hash ^= @bitCast(token);
            hash *%= 0x100000001b3;
            hash ^= hash >> 32;
        }
        return hash;
    }

    fn clearTable(self: *Proposer) void {
        @memset(&self.table, .{});
    }

    fn insert(self: *Proposer, hash: u64, start: usize) void {
        var slot: usize = @intCast(hash % capacity);
        // max_index_tokens <= capacity/2 bounds occupancy and probe work.
        for (0..capacity) |_| {
            const entry = &self.table[slot];
            if (!entry.occupied or entry.hash == hash) {
                entry.* = .{ .occupied = true, .hash = hash, .start = start };
                return;
            }
            slot = (slot + 1) % capacity;
        }
    }

    fn find(self: *const Proposer, hash: u64) ?usize {
        var slot: usize = @intCast(hash % capacity);
        for (0..capacity) |_| {
            const entry = self.table[slot];
            if (!entry.occupied) return null;
            if (entry.hash == hash) return entry.start;
            slot = (slot + 1) % capacity;
        }
        return null;
    }

    /// `history` must be the current request prompt plus its accepted generated
    /// prefix. Returned tokens are candidates only; the target remains the
    /// sole acceptance authority.
    pub fn propose(self: *Proposer, history: []const i64, remaining_budget: usize, output: []i64) !Proposal {
        self.counters.lookup_attempts +|= 1;
        const draft_cap = @min(@min(self.config.max_draft, remaining_budget), output.len);
        const n = self.config.min_ngram;
        if (draft_cap == 0 or history.len <= n) return self.noHit(output);

        self.clearTable();
        const suffix_start = history.len - n;
        const scan_start = suffix_start - @min(suffix_start, self.config.max_index_tokens);
        var start = scan_start;
        while (start + n < history.len) : (start += 1) {
            self.insert(tokenHash(history[start .. start + n]), start);
        }
        const suffix = history[suffix_start..];
        var source_start = self.find(tokenHash(suffix)) orelse return self.noHit(output);
        if (!std.mem.eql(i64, suffix, history[source_start .. source_start + n])) return self.noHit(output);

        // One index build per round: extend the verified match backward rather
        // than rebuilding a table for every candidate n-gram length.
        var matched = n;
        var query_start = suffix_start;
        while (matched < self.config.max_ngram and source_start > scan_start and query_start > 0 and
            history[source_start - 1] == history[query_start - 1])
        {
            source_start -= 1;
            query_start -= 1;
            matched += 1;
        }
        const continuation_start = source_start + matched;
        const count = @min(draft_cap, history.len - continuation_start);
        if (count == 0) return self.noHit(output);
        @memcpy(output[0..count], history[continuation_start..][0..count]);
        self.counters.lookup_hits +|= 1;
        self.counters.proposed_tokens +|= count;
        return .{ .tokens = output[0..count], .ngram_len = matched, .source_start = source_start };
    }

    fn noHit(self: *Proposer, output: []i64) Proposal {
        self.counters.zero_hit_fallbacks +|= 1;
        return .{ .tokens = output[0..0], .ngram_len = 0, .source_start = 0 };
    }

    pub fn noteVerification(self: *Proposer, proposed: usize, matched_drafts: usize, verify_rows: usize) void {
        self.counters.accepted_proposal_tokens +|= matched_drafts;
        self.counters.target_verify_passes +|= 1;
        self.counters.target_verify_rows +|= verify_rows;
        self.counters.rollback_tokens +|= proposed - @min(proposed, matched_drafts);
    }
};

test "copies a bounded continuation from current request history" {
    var proposer = try Proposer.init(.{ .min_ngram = 3, .max_ngram = 5, .max_draft = 4 });
    const history = [_]i64{ 90, 1, 2, 3, 7, 8, 9, 1, 2, 3 };
    var output: [16]i64 = undefined;
    const result = try proposer.propose(&history, 10, &output);
    try std.testing.expectEqualSlices(i64, &.{ 7, 8, 9, 1 }, result.tokens);
}

test "verified backward extension finds the longest local match" {
    var proposer = try Proposer.init(.{ .min_ngram = 2, .max_ngram = 4, .max_draft = 8 });
    const history = [_]i64{ 4, 5, 6, 7, 20, 21, 4, 5, 6, 7 };
    var output: [16]i64 = undefined;
    const result = try proposer.propose(&history, 1, &output);
    try std.testing.expectEqual(@as(usize, 4), result.ngram_len);
    try std.testing.expectEqualSlices(i64, &.{20}, result.tokens);
}

test "bounded index cannot see an expired prefix" {
    var proposer = try Proposer.init(.{ .min_ngram = 3, .max_ngram = 3, .max_draft = 4, .max_index_tokens = 4 });
    const history = [_]i64{ 1, 2, 3, 99, 70, 71, 72, 73, 1, 2, 3 };
    var output: [16]i64 = undefined;
    try std.testing.expectEqual(@as(usize, 0), (try proposer.propose(&history, 4, &output)).tokens.len);
}

test "verification accounting excludes correction and bonus" {
    var proposer = try Proposer.init(.{});
    proposer.noteVerification(6, 2, 7);
    try std.testing.expectEqual(@as(u64, 2), proposer.counters.accepted_proposal_tokens);
    try std.testing.expectEqual(@as(u64, 4), proposer.counters.rollback_tokens);
}

test "configuration hard bounds fail closed" {
    try std.testing.expectError(error.InvalidPromptLookupConfig, Proposer.init(.{ .max_ngram = 33 }));
    try std.testing.expectError(error.InvalidPromptLookupConfig, Proposer.init(.{ .max_index_tokens = 4097 }));
    try std.testing.expectError(error.InvalidPromptLookupConfig, Proposer.init(.{ .max_draft = 17 }));
}
