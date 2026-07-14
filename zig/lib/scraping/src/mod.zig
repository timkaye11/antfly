// Copyright 2026 Antfly, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const std = @import("std");
const objectstore = @import("objectstore");

const Allocator = std.mem.Allocator;
const default_max_download_size_bytes: u64 = 100 * 1024 * 1024;

pub const DownloadedContent = struct {
    content_type: []u8,
    data: []u8,

    pub fn deinit(self: *DownloadedContent, alloc: Allocator) void {
        alloc.free(self.content_type);
        alloc.free(self.data);
        self.* = undefined;
    }
};

pub const HttpError = struct {
    status: u16,
    message: []const u8,
};

pub const DownloadOutcome = union(enum) {
    ok: DownloadedContent,
    http_error: HttpError,
};

pub const HTTPHeader = struct {
    name: []const u8,
    value: []const u8,
};

pub const ContentSecurityConfig = struct {
    allowed_hosts: ?[]const []u8 = null,
    block_private_ips: ?bool = null,
    max_download_size_bytes: ?u64 = null,
    download_timeout_seconds: ?u32 = null,
    max_image_dimension: ?u32 = null,
    allowed_paths: ?[]const []u8 = null,
    user_agent: ?[]u8 = null,

    pub fn deinit(self: *ContentSecurityConfig, alloc: std.mem.Allocator) void {
        if (self.allowed_hosts) |values| freeOwnedStringSlice(alloc, values);
        if (self.allowed_paths) |values| freeOwnedStringSlice(alloc, values);
        if (self.user_agent) |value| alloc.free(value);
        self.* = undefined;
    }
};

pub const S3CredentialsConfig = struct {
    endpoint: ?[]u8 = null,
    use_ssl: ?bool = null,
    access_key_id: ?[]u8 = null,
    secret_access_key: ?[]u8 = null,
    session_token: ?[]u8 = null,

    pub fn deinit(self: *S3CredentialsConfig, alloc: std.mem.Allocator) void {
        if (self.endpoint) |value| alloc.free(value);
        if (self.access_key_id) |value| alloc.free(value);
        if (self.secret_access_key) |value| alloc.free(value);
        if (self.session_token) |value| alloc.free(value);
        self.* = undefined;
    }
};

pub const S3CredentialConfig = struct {
    endpoint: ?[]u8 = null,
    use_ssl: ?bool = null,
    access_key_id: ?[]u8 = null,
    secret_access_key: ?[]u8 = null,
    session_token: ?[]u8 = null,
    buckets: ?[]const []u8 = null,
    security: ?ContentSecurityConfig = null,

    pub fn deinit(self: *S3CredentialConfig, alloc: std.mem.Allocator) void {
        if (self.endpoint) |value| alloc.free(value);
        if (self.access_key_id) |value| alloc.free(value);
        if (self.secret_access_key) |value| alloc.free(value);
        if (self.session_token) |value| alloc.free(value);
        if (self.buckets) |values| freeOwnedStringSlice(alloc, values);
        if (self.security) |*security| security.deinit(alloc);
        self.* = undefined;
    }
};

pub const HTTPCredentialConfig = struct {
    base_url: ?[]u8 = null,
    headers: std.StringArrayHashMapUnmanaged([]u8) = .{},
    security: ?ContentSecurityConfig = null,

    pub fn deinit(self: *HTTPCredentialConfig, alloc: std.mem.Allocator) void {
        if (self.base_url) |value| alloc.free(value);
        var it = self.headers.iterator();
        while (it.next()) |entry| {
            alloc.free(entry.key_ptr.*);
            alloc.free(entry.value_ptr.*);
        }
        self.headers.deinit(alloc);
        if (self.security) |*security| security.deinit(alloc);
        self.* = undefined;
    }
};

pub const RemoteContentConfig = struct {
    security: ?ContentSecurityConfig = null,
    default_s3: ?[]u8 = null,
    s3: std.StringArrayHashMapUnmanaged(S3CredentialConfig) = .{},
    http: std.StringArrayHashMapUnmanaged(HTTPCredentialConfig) = .{},

    pub fn getS3(self: *const RemoteContentConfig, name: []const u8) ?*const S3CredentialConfig {
        return self.s3.getPtr(name);
    }

    pub fn getHttp(self: *const RemoteContentConfig, name: []const u8) ?*const HTTPCredentialConfig {
        return self.http.getPtr(name);
    }

    pub fn deinit(self: *RemoteContentConfig, alloc: std.mem.Allocator) void {
        if (self.security) |*security| security.deinit(alloc);
        if (self.default_s3) |value| alloc.free(value);

        var s3_it = self.s3.iterator();
        while (s3_it.next()) |entry| {
            alloc.free(entry.key_ptr.*);
            entry.value_ptr.deinit(alloc);
        }
        self.s3.deinit(alloc);

        var http_it = self.http.iterator();
        while (http_it.next()) |entry| {
            alloc.free(entry.key_ptr.*);
            entry.value_ptr.deinit(alloc);
        }
        self.http.deinit(alloc);
        self.* = undefined;
    }
};

pub fn downloadContentAlloc(
    alloc: Allocator,
    uri: []const u8,
    security: ?*const ContentSecurityConfig,
    s3_credentials: ?*const S3CredentialsConfig,
) !DownloadedContent {
    const outcome = try downloadContentOutcomeAlloc(alloc, uri, security, s3_credentials);
    return switch (outcome) {
        .ok => |downloaded| downloaded,
        .http_error => error.HttpFetchFailed,
    };
}

pub fn downloadContentOutcomeAlloc(
    alloc: Allocator,
    uri: []const u8,
    security: ?*const ContentSecurityConfig,
    s3_credentials: ?*const S3CredentialsConfig,
) !DownloadOutcome {
    return try downloadContentOutcomeAllocWithHeaders(alloc, uri, security, s3_credentials, null);
}

pub fn downloadContentOutcomeAllocWithHeaders(
    alloc: Allocator,
    uri: []const u8,
    security: ?*const ContentSecurityConfig,
    s3_credentials: ?*const S3CredentialsConfig,
    http_headers: ?[]const HTTPHeader,
) !DownloadOutcome {
    if (std.mem.startsWith(u8, uri, "data:")) {
        return .{ .ok = try parseDataUriAlloc(alloc, uri, security) };
    }

    const parsed = try std.Uri.parse(uri);
    if (std.mem.eql(u8, parsed.scheme, "http") or std.mem.eql(u8, parsed.scheme, "https")) {
        try validateUrlSecurity(parsed, security);
        return try downloadHttpOutcomeAlloc(alloc, parsed, security, http_headers);
    }
    if (std.mem.eql(u8, parsed.scheme, "file")) {
        const path_buf = try alloc.dupe(u8, parsed.path.percent_encoded);
        defer alloc.free(path_buf);
        const path = std.Uri.percentDecodeInPlace(path_buf);
        return .{ .ok = try downloadFileAlloc(alloc, path, security) };
    }
    if (std.mem.eql(u8, parsed.scheme, "s3")) {
        return .{ .ok = try downloadS3Alloc(alloc, parsed, security, s3_credentials) };
    }
    return error.UnsupportedUrlScheme;
}

pub fn isEmptyContentSecurity(value: ContentSecurityConfig) bool {
    return value.allowed_hosts == null and
        value.block_private_ips == null and
        value.max_download_size_bytes == null and
        value.download_timeout_seconds == null and
        value.max_image_dimension == null and
        value.allowed_paths == null and
        value.user_agent == null;
}

pub fn effectiveContentSecurity(
    primary: ?*const ContentSecurityConfig,
    fallback: ?*const ContentSecurityConfig,
) ?*const ContentSecurityConfig {
    if (primary) |security| {
        if (!isEmptyContentSecurity(security.*)) return security;
    }
    if (fallback) |security| {
        if (!isEmptyContentSecurity(security.*)) return security;
    }
    return null;
}

fn freeOwnedStringSlice(alloc: std.mem.Allocator, values: []const []u8) void {
    for (values) |value| alloc.free(value);
    alloc.free(values);
}

pub fn dataUriDecodedSize(uri: []const u8) !usize {
    const prefix = "data:";
    if (!std.mem.startsWith(u8, uri, prefix)) return error.InvalidDataUri;

    const payload = uri[prefix.len..];
    const comma = std.mem.indexOfScalar(u8, payload, ',') orelse return error.InvalidDataUri;
    const meta = payload[0..comma];
    const body = payload[comma + 1 ..];

    if (std.mem.endsWith(u8, meta, ";base64")) {
        return std.base64.standard.Decoder.calcSizeForSlice(body) catch return error.InvalidBase64;
    }

    return try percentDecodedLen(body);
}

fn percentDecodedLen(value: []const u8) !usize {
    var len: usize = 0;
    var i: usize = 0;
    while (i < value.len) {
        if (value[i] == '%') {
            if (i + 2 >= value.len) return error.InvalidDataUri;
            _ = std.fmt.charToDigit(value[i + 1], 16) catch return error.InvalidDataUri;
            _ = std.fmt.charToDigit(value[i + 2], 16) catch return error.InvalidDataUri;
            i += 3;
        } else {
            i += 1;
        }
        len += 1;
    }
    return len;
}

fn validateDownloadSize(decoded_len: usize, security: ?*const ContentSecurityConfig) !void {
    if (@as(u64, @intCast(decoded_len)) > maxDownloadSizeBytes(security)) return error.StreamTooLong;
}

fn maxDownloadSizeBytes(security: ?*const ContentSecurityConfig) u64 {
    return if (security) |cfg|
        cfg.max_download_size_bytes orelse default_max_download_size_bytes
    else
        default_max_download_size_bytes;
}

fn maxDownloadSize(security: ?*const ContentSecurityConfig) usize {
    return std.math.cast(usize, maxDownloadSizeBytes(security)) orelse std.math.maxInt(usize);
}

fn parseDataUriAlloc(alloc: Allocator, uri: []const u8, security: ?*const ContentSecurityConfig) !DownloadedContent {
    const prefix = "data:";
    if (!std.mem.startsWith(u8, uri, prefix)) return error.InvalidDataUri;

    const payload = uri[prefix.len..];
    const comma = std.mem.indexOfScalar(u8, payload, ',') orelse return error.InvalidDataUri;
    const meta = payload[0..comma];
    const body = payload[comma + 1 ..];

    if (std.mem.endsWith(u8, meta, ";base64")) {
        const mime = meta[0 .. meta.len - ";base64".len];
        const decoded_len = std.base64.standard.Decoder.calcSizeForSlice(body) catch return error.InvalidBase64;
        try validateDownloadSize(decoded_len, security);
        const decoded = try alloc.alloc(u8, decoded_len);
        errdefer alloc.free(decoded);
        std.base64.standard.Decoder.decode(decoded, body) catch return error.InvalidBase64;
        return .{
            .content_type = try alloc.dupe(u8, if (mime.len > 0) mime else "application/octet-stream"),
            .data = decoded,
        };
    }

    const decoded_len = try percentDecodedLen(body);
    try validateDownloadSize(decoded_len, security);
    const decoded_body_buf = try alloc.dupe(u8, body);
    var data = decoded_body_buf;
    errdefer alloc.free(data);
    const decoded_body = std.Uri.percentDecodeInPlace(decoded_body_buf);
    if (decoded_body.len != decoded_body_buf.len) {
        const exact = try alloc.dupe(u8, decoded_body);
        alloc.free(decoded_body_buf);
        data = exact;
    }

    return .{
        .content_type = try alloc.dupe(u8, if (meta.len > 0) meta else "text/plain"),
        .data = data,
    };
}

fn downloadHttpOutcomeAlloc(
    alloc: Allocator,
    uri: std.Uri,
    security: ?*const ContentSecurityConfig,
    http_headers: ?[]const HTTPHeader,
) !DownloadOutcome {
    var io_impl = std.Io.Threaded.init(alloc, .{});
    defer io_impl.deinit();

    var client = std.http.Client{
        .allocator = alloc,
        .io = io_impl.io(),
    };
    defer client.deinit();

    var headers = std.ArrayListUnmanaged(std.http.Header).empty;
    defer headers.deinit(alloc);
    try headers.append(alloc, .{
        .name = "User-Agent",
        .value = if (security) |cfg| cfg.user_agent orelse "AntflyDB/1.0" else "AntflyDB/1.0",
    });
    if (http_headers) |extra_headers| {
        for (extra_headers) |header| {
            if (header.name.len == 0) continue;
            try headers.append(alloc, .{ .name = header.name, .value = header.value });
        }
    }

    var request = try std.http.Client.request(&client, .GET, uri, httpRequestOptions(headers.items));
    defer request.deinit();

    try request.sendBodiless();
    var response = try request.receiveHead(&.{});
    if (response.head.status.class() != .success) {
        return .{
            .http_error = .{
                .status = @intFromEnum(response.head.status),
                .message = "remote fetch failed",
            },
        };
    }

    const mime = if (response.head.content_type) |value|
        trimMimeParameters(value)
    else
        "application/octet-stream";

    const max_size = maxDownloadSize(security);

    var transfer_buffer: [512]u8 = undefined;
    const body = try response.reader(&transfer_buffer).allocRemaining(alloc, .limited(max_size));
    errdefer alloc.free(body);

    return .{ .ok = .{
        .content_type = try alloc.dupe(u8, mime),
        .data = body,
    } };
}

fn httpRequestOptions(extra_headers: []const std.http.Header) std.http.Client.RequestOptions {
    return .{
        .keep_alive = false,
        // Redirect targets must be parsed and revalidated before any follow-up
        // request. This downloader intentionally returns 3xx as a bounded HTTP
        // error instead of relying on an empty redirect buffer to stop follows.
        .redirect_behavior = .unhandled,
        .extra_headers = extra_headers,
    };
}

fn downloadFileAlloc(
    alloc: Allocator,
    path: []const u8,
    security: ?*const ContentSecurityConfig,
) !DownloadedContent {
    if (security) |cfg| {
        if (cfg.allowed_paths) |allowed_paths| {
            if (allowed_paths.len == 0) return error.PathNotAllowed;
        }
    }

    var io_impl = std.Io.Threaded.init(alloc, .{});
    defer io_impl.deinit();

    const limit = maxDownloadSize(security);
    try validateFilePathSecurityBeforeOpen(alloc, io_impl.io(), path, security);
    var file = try std.Io.Dir.openFileAbsolute(io_impl.io(), path, .{});
    defer file.close(io_impl.io());
    try validateOpenFilePathSecurity(alloc, io_impl.io(), file, security);
    var reader = file.reader(io_impl.io(), &.{});
    const data = try reader.interface.allocRemaining(alloc, .limited(limit));
    errdefer alloc.free(data);
    return .{
        .content_type = try alloc.dupe(u8, guessMimeType(path)),
        .data = data,
    };
}

fn downloadS3Alloc(
    alloc: Allocator,
    parsed: std.Uri,
    security: ?*const ContentSecurityConfig,
    s3_credentials: ?*const S3CredentialsConfig,
) !DownloadedContent {
    const creds_cfg = s3_credentials orelse return error.MissingS3Credentials;
    const configured_endpoint = creds_cfg.endpoint orelse return error.MissingEndpoint;
    var resolved_endpoint = try objectstore.resolveS3EndpointAlloc(
        alloc,
        configured_endpoint,
        creds_cfg.use_ssl orelse true,
    );
    defer resolved_endpoint.deinit(alloc);

    const bucket, const key = try parseS3LocationAlloc(
        alloc,
        parsed,
        resolved_endpoint.endpoint,
        resolved_endpoint.use_ssl,
    );
    defer alloc.free(bucket);
    defer alloc.free(key);

    const joined_path = try std.fmt.allocPrint(alloc, "{s}/{s}", .{ bucket, key });
    defer alloc.free(joined_path);
    try validateS3PathSecurity(joined_path, security);

    var creds = try allocS3Credentials(alloc, resolved_endpoint.endpoint, resolved_endpoint.use_ssl, creds_cfg);
    var client = objectstore.S3.Client.init(alloc, .{
        .credentials = creds,
        .addressing_style = .path,
    }) catch |err| {
        creds.deinit(alloc);
        return err;
    };
    defer client.deinit();

    var store_client = client.client();
    const max_size = maxDownloadSizeBytes(security);
    var metadata = try store_client.statObject(bucket, key);
    defer metadata.deinit(alloc);
    if (metadata.content_length > max_size) return error.StreamTooLong;

    const range_length = std.math.add(u64, max_size, 1) catch null;
    var result = try store_client.getObject(bucket, key, .{
        .range = if (range_length) |length| .{ .offset = 0, .length = length } else null,
    });
    defer result.deinit(alloc);
    if (result.body.len > max_size) return error.StreamTooLong;

    return .{
        .content_type = try alloc.dupe(u8, result.metadata.content_type orelse guessMimeType(key)),
        .data = try alloc.dupe(u8, result.body),
    };
}

fn allocS3Credentials(
    alloc: Allocator,
    endpoint: []const u8,
    use_ssl: bool,
    cfg: *const S3CredentialsConfig,
) !objectstore.S3Credentials {
    const access_key_id = cfg.access_key_id orelse return error.MissingAccessKeyId;
    const secret_access_key = cfg.secret_access_key orelse return error.MissingSecretAccessKey;
    const owned_endpoint = try alloc.dupe(u8, endpoint);
    errdefer alloc.free(owned_endpoint);
    const owned_access_key_id = try alloc.dupe(u8, access_key_id);
    errdefer alloc.free(owned_access_key_id);
    const owned_secret_access_key = try alloc.dupe(u8, secret_access_key);
    errdefer alloc.free(owned_secret_access_key);
    const owned_session_token = if (cfg.session_token) |value| try alloc.dupe(u8, value) else null;
    errdefer if (owned_session_token) |value| alloc.free(value);
    const region = try alloc.dupe(u8, "us-east-1");
    errdefer alloc.free(region);
    return .{
        .endpoint = owned_endpoint,
        .use_ssl = use_ssl,
        .access_key_id = owned_access_key_id,
        .secret_access_key = owned_secret_access_key,
        .session_token = owned_session_token,
        .region = region,
    };
}

fn parseS3LocationAlloc(
    alloc: Allocator,
    parsed: std.Uri,
    configured_endpoint: []const u8,
    use_ssl: bool,
) !struct { []u8, []u8 } {
    const host = parsed.host orelse return error.InvalidS3Url;
    const host_text = host.percent_encoded;
    const path = trimLeftSlash(parsed.path.percent_encoded);
    if (path.len == 0) return error.InvalidS3Url;

    if (try s3AuthorityMatchesEndpoint(alloc, host_text, parsed.port, configured_endpoint, use_ssl)) {
        const slash = std.mem.indexOfScalar(u8, path, '/') orelse return error.InvalidS3Url;
        return .{
            try alloc.dupe(u8, path[0..slash]),
            try alloc.dupe(u8, path[slash + 1 ..]),
        };
    }

    return .{
        try alloc.dupe(u8, host_text),
        try alloc.dupe(u8, path),
    };
}

fn s3AuthorityMatchesEndpoint(
    alloc: Allocator,
    host: []const u8,
    port: ?u16,
    endpoint: []const u8,
    use_ssl: bool,
) !bool {
    const authority = if (port) |value|
        try std.fmt.allocPrint(alloc, "{s}:{d}", .{ host, value })
    else
        try alloc.dupe(u8, host);
    defer alloc.free(authority);
    if (std.ascii.eqlIgnoreCase(authority, endpoint)) return true;

    const default_port: u16 = if (use_ssl) 443 else 80;
    if (port == default_port and std.ascii.eqlIgnoreCase(host, endpoint)) return true;
    if (port == null) {
        const endpoint_with_default_port = try std.fmt.allocPrint(alloc, "{s}:{d}", .{ host, default_port });
        defer alloc.free(endpoint_with_default_port);
        if (std.ascii.eqlIgnoreCase(endpoint_with_default_port, endpoint)) return true;
    }
    return false;
}

fn validateUrlSecurity(parsed: std.Uri, security: ?*const ContentSecurityConfig) !void {
    const cfg = security orelse return;
    var host_buffer: [std.Io.net.HostName.max_len]u8 = undefined;
    const host = (parsed.getHost(&host_buffer) catch return error.InvalidHost).bytes;

    if (cfg.allowed_hosts) |allowed_hosts| {
        var allowed = false;
        for (allowed_hosts) |entry| {
            if (std.ascii.eqlIgnoreCase(entry, host)) {
                allowed = true;
                break;
            }
        }
        if (!allowed) return error.HostNotAllowed;
    }

    if (cfg.block_private_ips orelse true) {
        // std.http resolves hostnames internally, so this layer cannot verify
        // and pin the address it ultimately connects to. Fail closed for DNS
        // names until the transport exposes that guarantee; operators can use
        // the existing explicit `block_private_ips = false` opt-out.
        if (!isCanonicalIpLiteral(host) or isNonGlobalHost(host)) return error.PrivateIpBlocked;
    }
}

fn validateOpenFilePathSecurity(
    alloc: Allocator,
    io: std.Io,
    file: std.Io.File,
    security: ?*const ContentSecurityConfig,
) !void {
    const cfg = security orelse return;
    const allowed_paths = cfg.allowed_paths orelse return;
    var canonical_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const canonical_path = canonical_path_buf[0..try file.realPath(io, &canonical_path_buf)];
    try validateCanonicalFilePathSecurity(alloc, io, canonical_path, allowed_paths);
}

fn validateFilePathSecurityBeforeOpen(
    alloc: Allocator,
    io: std.Io,
    path: []const u8,
    security: ?*const ContentSecurityConfig,
) !void {
    const cfg = security orelse return;
    const allowed_paths = cfg.allowed_paths orelse return;
    const resolved_path = try lexicalAbsolutePathAlloc(alloc, io, path);
    defer alloc.free(resolved_path);
    for (allowed_paths) |allowed| {
        const resolved_allowed = try lexicalAbsolutePathAlloc(alloc, io, allowed);
        defer alloc.free(resolved_allowed);
        if (hasComponentPrefix(resolved_path, resolved_allowed, std.fs.path.sep)) return;
    }
    return error.PathNotAllowed;
}

fn lexicalAbsolutePathAlloc(alloc: Allocator, io: std.Io, path: []const u8) ![]u8 {
    if (std.fs.path.isAbsolute(path)) return std.fs.path.resolve(alloc, &.{path});
    const cwd = try std.Io.Dir.cwd().realPathFileAlloc(io, ".", alloc);
    defer alloc.free(cwd);
    return std.fs.path.resolve(alloc, &.{ cwd, path });
}

fn validateCanonicalFilePathSecurity(
    alloc: Allocator,
    io: std.Io,
    canonical_path: []const u8,
    allowed_paths: []const []u8,
) !void {
    for (allowed_paths) |allowed| {
        const canonical_allowed = realPathExistingAlloc(alloc, io, allowed) catch |err| switch (err) {
            error.FileNotFound, error.NotDir => continue,
            else => return err,
        };
        defer alloc.free(canonical_allowed);
        if (hasComponentPrefix(canonical_path, canonical_allowed, std.fs.path.sep)) return;
    }
    return error.PathNotAllowed;
}

fn validateS3PathSecurity(path: []const u8, security: ?*const ContentSecurityConfig) !void {
    const cfg = security orelse return;
    const allowed_paths = cfg.allowed_paths orelse return;
    for (allowed_paths) |allowed| {
        if (hasComponentPrefix(path, allowed, '/')) return;
    }
    return error.PathNotAllowed;
}

fn realPathExistingAlloc(alloc: Allocator, io: std.Io, path: []const u8) ![:0]u8 {
    if (std.fs.path.isAbsolute(path)) {
        return try std.Io.Dir.realPathFileAbsoluteAlloc(io, path, alloc);
    }
    return try std.Io.Dir.cwd().realPathFileAlloc(io, path, alloc);
}

fn hasComponentPrefix(path: []const u8, prefix: []const u8, separator: u8) bool {
    if (prefix.len == 0 or !std.mem.startsWith(u8, path, prefix)) return false;
    return path.len == prefix.len or prefix[prefix.len - 1] == separator or path[prefix.len] == separator;
}

fn isCanonicalIpLiteral(host: []const u8) bool {
    var literal = host;
    if (literal.len >= 2 and literal[0] == '[' and literal[literal.len - 1] == ']') {
        literal = literal[1 .. literal.len - 1];
    }
    if (std.mem.indexOfScalar(u8, literal, '%') != null) return false;
    _ = std.Io.net.IpAddress.parse(literal, 0) catch return false;
    return true;
}

/// `block_private_ips` rejects literal addresses that are not globally
/// reachable, not only RFC 1918 addresses. Safe-mode callers reject hostnames
/// before reaching this helper because std.http cannot pin a vetted DNS result.
fn isNonGlobalHost(host: []const u8) bool {
    var normalized = std.mem.trimEnd(u8, host, ".");
    if (normalized.len >= 2 and normalized[0] == '[' and normalized[normalized.len - 1] == ']') {
        normalized = normalized[1 .. normalized.len - 1];
    }
    // Zone identifiers are only meaningful for scoped destinations. Zig's
    // literal parser intentionally leaves them to the resolver, so reject
    // them here rather than accidentally treating a scoped address as public.
    if (std.mem.indexOfScalar(u8, normalized, '%') != null) return true;
    if (std.ascii.eqlIgnoreCase(normalized, "localhost")) return true;
    if (std.ascii.endsWithIgnoreCase(normalized, ".localhost")) return true;
    if (std.ascii.endsWithIgnoreCase(normalized, ".local")) return true;

    const address = std.Io.net.IpAddress.parse(normalized, 0) catch
        return looksLikeNoncanonicalIpLiteral(normalized);
    return switch (address) {
        .ip4 => |ip4| {
            return isNonGlobalIpv4(ip4.bytes);
        },
        .ip6 => |ip6| {
            return !ip6.interface.isNone() or isNonGlobalIpv6(ip6.bytes);
        },
    };
}

fn looksLikeNoncanonicalIpLiteral(host: []const u8) bool {
    if (host.len == 0) return false;

    var decimal_only = true;
    for (host) |char| {
        if (!std.ascii.isDigit(char) and char != '.') {
            decimal_only = false;
            break;
        }
    }
    if (decimal_only) return true;

    var saw_hex_component = false;
    var components = std.mem.splitScalar(u8, host, '.');
    while (components.next()) |component| {
        if (component.len > 2 and component[0] == '0' and (component[1] == 'x' or component[1] == 'X')) {
            for (component[2..]) |char| if (!std.ascii.isHex(char)) return false;
            saw_hex_component = true;
            continue;
        }
        if (component.len == 0) return false;
        for (component) |char| if (!std.ascii.isDigit(char)) return false;
    }
    return saw_hex_component;
}

fn isNonGlobalIpv4(b: [4]u8) bool {
    return b[0] == 0 or
        b[0] == 10 or
        (b[0] == 100 and b[1] >= 64 and b[1] <= 127) or
        b[0] == 127 or
        (b[0] == 169 and b[1] == 254) or
        (b[0] == 172 and b[1] >= 16 and b[1] <= 31) or
        (b[0] == 192 and b[1] == 0 and (b[2] == 0 or b[2] == 2)) or
        (b[0] == 192 and b[1] == 168) or
        (b[0] == 192 and b[1] == 88 and b[2] == 99) or
        (b[0] == 198 and (b[1] == 18 or b[1] == 19)) or
        (b[0] == 198 and b[1] == 51 and b[2] == 100) or
        (b[0] == 203 and b[1] == 0 and b[2] == 113) or
        b[0] >= 224;
}

fn isNonGlobalIpv6(b: [16]u8) bool {
    // Preserve the IPv4 classification for IPv4-mapped literals.
    if (std.mem.allEqual(u8, b[0..10], 0) and b[10] == 0xff and b[11] == 0xff) {
        return isNonGlobalIpv4(b[12..16].*);
    }

    // Transition addresses carry an IPv4 destination and must inherit its
    // classification instead of treating the outer IPv6 prefix as public.
    if (b[0] == 0 and b[1] == 0x64 and b[2] == 0xff and b[3] == 0x9b and
        std.mem.allEqual(u8, b[4..12], 0))
    {
        return isNonGlobalIpv4(b[12..16].*);
    }
    if (b[0] == 0x20 and b[1] == 0x02) return isNonGlobalIpv4(b[2..6].*);

    // Conservatively reject space outside global unicast (2000::/3). This
    // includes the RFC 8215 local-use NAT64 prefix 64:ff9b:1::/48. Then reject
    // special-use sub-ranges that sit inside global unicast, including Teredo:
    // its outer address is global-looking but embeds an IPv4 destination.
    if ((b[0] & 0xe0) != 0x20) return true;
    return (b[0] == 0x20 and b[1] == 0x01 and b[2] == 0 and b[3] == 0) or // Teredo
        (b[0] == 0x20 and b[1] == 0x01 and b[2] == 0 and b[3] == 0x02 and b[4] == 0 and b[5] == 0) or // benchmarking
        (b[0] == 0x20 and b[1] == 0x01 and b[2] == 0 and (b[3] & 0xf0) == 0x10) or // ORCHID
        (b[0] == 0x20 and b[1] == 0x01 and b[2] == 0 and (b[3] & 0xf0) == 0x20) or // ORCHIDv2
        (b[0] == 0x20 and b[1] == 0x01 and b[2] == 0x0d and b[3] == 0xb8) or // documentation
        (b[0] == 0x3f and b[1] == 0xfe) or // deprecated 6bone
        (b[0] == 0x3f and b[1] == 0xff and (b[2] & 0xf0) == 0); // documentation
}

fn guessMimeType(path: []const u8) []const u8 {
    const ext = std.fs.path.extension(path);
    if (std.ascii.eqlIgnoreCase(ext, ".html") or std.ascii.eqlIgnoreCase(ext, ".htm")) return "text/html";
    if (std.ascii.eqlIgnoreCase(ext, ".pdf")) return "application/pdf";
    if (std.ascii.eqlIgnoreCase(ext, ".png")) return "image/png";
    if (std.ascii.eqlIgnoreCase(ext, ".jpg") or std.ascii.eqlIgnoreCase(ext, ".jpeg")) return "image/jpeg";
    if (std.ascii.eqlIgnoreCase(ext, ".gif")) return "image/gif";
    if (std.ascii.eqlIgnoreCase(ext, ".webp")) return "image/webp";
    if (std.ascii.eqlIgnoreCase(ext, ".svg")) return "image/svg+xml";
    if (std.ascii.eqlIgnoreCase(ext, ".txt")) return "text/plain";
    if (std.ascii.eqlIgnoreCase(ext, ".md") or std.ascii.eqlIgnoreCase(ext, ".markdown")) return "text/markdown";
    return "application/octet-stream";
}

fn trimLeftSlash(path: []const u8) []const u8 {
    var idx: usize = 0;
    while (idx < path.len and path[idx] == '/') : (idx += 1) {}
    return path[idx..];
}

fn trimMimeParameters(value: []const u8) []const u8 {
    const semi = std.mem.indexOfScalar(u8, value, ';') orelse return value;
    return std.mem.trim(u8, value[0..semi], &std.ascii.whitespace);
}

test "effective content security prefers primary when non-empty" {
    var primary = ContentSecurityConfig{
        .block_private_ips = false,
    };
    var fallback = ContentSecurityConfig{
        .block_private_ips = true,
    };
    const effective = effectiveContentSecurity(&primary, &fallback).?;
    try std.testing.expectEqual(@as(?bool, false), effective.block_private_ips);
}

test "effective content security falls back when primary is empty" {
    var primary = ContentSecurityConfig{};
    var fallback = ContentSecurityConfig{
        .block_private_ips = true,
    };
    const effective = effectiveContentSecurity(&primary, &fallback).?;
    try std.testing.expectEqual(@as(?bool, true), effective.block_private_ips);
}

test "HTTP downloads leave redirects unhandled for target revalidation" {
    const headers = [_]std.http.Header{.{ .name = "User-Agent", .value = "test" }};
    const options = httpRequestOptions(&headers);
    try std.testing.expectEqual(std.http.Client.Request.RedirectBehavior.unhandled, options.redirect_behavior);
    try std.testing.expect(!options.keep_alive);
    try std.testing.expectEqual(@as(usize, 1), options.extra_headers.len);
}

test "download content parses data uri" {
    const alloc = std.testing.allocator;
    var downloaded = try downloadContentAlloc(alloc, "data:text/plain;base64,aGVsbG8=", null, null);
    defer downloaded.deinit(alloc);
    try std.testing.expectEqualStrings("text/plain", downloaded.content_type);
    try std.testing.expectEqualStrings("hello", downloaded.data);
}

test "download content percent decodes non-base64 data uri" {
    const alloc = std.testing.allocator;
    var downloaded = try downloadContentAlloc(alloc, "data:text/plain,alpha%20beta%2Bgamma", null, null);
    defer downloaded.deinit(alloc);
    try std.testing.expectEqualStrings("text/plain", downloaded.content_type);
    try std.testing.expectEqualStrings("alpha beta+gamma", downloaded.data);
}

test "download content enforces data uri decoded byte limit" {
    const alloc = std.testing.allocator;
    var security = ContentSecurityConfig{ .max_download_size_bytes = 4 };
    try std.testing.expectError(error.StreamTooLong, downloadContentAlloc(alloc, "data:text/plain;base64,aGVsbG8=", &security, null));
    try std.testing.expectError(error.StreamTooLong, downloadContentAlloc(alloc, "data:text/plain,alpha%20beta", &security, null));
}

test "download size defaults apply when content security omits the field" {
    const security = ContentSecurityConfig{};
    try std.testing.expectEqual(default_max_download_size_bytes, maxDownloadSizeBytes(null));
    try std.testing.expectEqual(default_max_download_size_bytes, maxDownloadSizeBytes(&security));
    try std.testing.expectError(error.StreamTooLong, validateDownloadSize(default_max_download_size_bytes + 1, &security));
}

test "download content reads percent encoded file uri" {
    const alloc = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "image file.png",
        .data = "png-bytes",
    });

    const rel_path = try std.fs.path.join(alloc, &.{ ".zig-cache", "tmp", tmp.sub_path[0..], "image file.png" });
    defer alloc.free(rel_path);
    const abs_path = try std.Io.Dir.cwd().realPathFileAlloc(std.testing.io, rel_path, alloc);
    defer alloc.free(abs_path);

    const raw_uri = try std.fmt.allocPrint(alloc, "file://{s}", .{abs_path});
    defer alloc.free(raw_uri);
    const uri = try std.mem.replaceOwned(u8, alloc, raw_uri, " ", "%20");
    defer alloc.free(uri);

    var downloaded = try downloadContentAlloc(alloc, uri, null, null);
    defer downloaded.deinit(alloc);
    try std.testing.expectEqualStrings("image/png", downloaded.content_type);
    try std.testing.expectEqualStrings("png-bytes", downloaded.data);
}

test "file path security requires canonical component containment" {
    const alloc = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.createDirPath(std.testing.io, "allowed");
    try tmp.dir.createDirPath(std.testing.io, "allowed-sibling");
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "allowed/child.txt", .data = "allowed" });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "allowed-sibling/child.txt", .data = "blocked" });

    const allowed_root = try tmp.dir.realPathFileAlloc(std.testing.io, "allowed", alloc);
    defer alloc.free(allowed_root);
    const allowed_child = try tmp.dir.realPathFileAlloc(std.testing.io, "allowed/child.txt", alloc);
    defer alloc.free(allowed_child);
    const sibling_child = try tmp.dir.realPathFileAlloc(std.testing.io, "allowed-sibling/child.txt", alloc);
    defer alloc.free(sibling_child);
    const allowed_paths = [_][]u8{allowed_root};
    const security = ContentSecurityConfig{ .allowed_paths = &allowed_paths };

    var downloaded = try downloadFileAlloc(alloc, allowed_child, &security);
    defer downloaded.deinit(alloc);
    try std.testing.expectEqualStrings("allowed", downloaded.data);
    try std.testing.expectError(error.PathNotAllowed, downloadFileAlloc(alloc, sibling_child, &security));
}

test "file path security accepts relative roots and rejects outside paths before lookup" {
    const alloc = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(std.testing.io, "allowed");
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "allowed/child.txt", .data = "allowed" });

    const relative_root = try std.fs.path.join(alloc, &.{ ".zig-cache", "tmp", tmp.sub_path[0..], "allowed" });
    defer alloc.free(relative_root);
    const absolute_child = try tmp.dir.realPathFileAlloc(std.testing.io, "allowed/child.txt", alloc);
    defer alloc.free(absolute_child);
    const allowed_paths = [_][]u8{relative_root};
    const security = ContentSecurityConfig{ .allowed_paths = &allowed_paths };

    var downloaded = try downloadFileAlloc(alloc, absolute_child, &security);
    defer downloaded.deinit(alloc);
    try std.testing.expectEqualStrings("allowed", downloaded.data);

    const nonexistent_allowed = [_][]u8{@constCast("/definitely/not/an/antfly/allowed/root")};
    try std.testing.expectError(error.PathNotAllowed, validateFilePathSecurityBeforeOpen(
        alloc,
        std.testing.io,
        "/definitely/not/an/antfly/outside/file",
        &.{ .allowed_paths = &nonexistent_allowed },
    ));
}

test "file path security rejects symlink escape" {
    const builtin = @import("builtin");
    if (builtin.os.tag == .windows or builtin.os.tag == .wasi or builtin.os.tag == .freestanding) return error.SkipZigTest;

    const alloc = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.createDirPath(std.testing.io, "allowed");
    try tmp.dir.createDirPath(std.testing.io, "outside");
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "outside/secret.txt", .data = "secret" });
    try tmp.dir.symLink(std.testing.io, "../outside/secret.txt", "allowed/link.txt", .{});

    const allowed_root = try tmp.dir.realPathFileAlloc(std.testing.io, "allowed", alloc);
    defer alloc.free(allowed_root);
    const link_path = try std.fs.path.join(alloc, &.{ allowed_root, "link.txt" });
    defer alloc.free(link_path);
    const allowed_paths = [_][]u8{allowed_root};
    const security = ContentSecurityConfig{ .allowed_paths = &allowed_paths };

    try std.testing.expectError(error.PathNotAllowed, downloadFileAlloc(alloc, link_path, &security));
}

test "explicit empty allowlists deny remote content without probing targets" {
    const allowed_hosts = [_][]u8{};
    const allowed_paths = [_][]u8{};
    const security = ContentSecurityConfig{
        .allowed_hosts = &allowed_hosts,
        .allowed_paths = &allowed_paths,
    };

    try std.testing.expectError(error.HostNotAllowed, downloadContentAlloc(std.testing.allocator, "https://does-not-exist.invalid/image.png", &security, null));
    try std.testing.expectError(error.PathNotAllowed, downloadFileAlloc(std.testing.allocator, "/path/that/does/not/exist", &security));
    try std.testing.expectError(error.PathNotAllowed, validateS3PathSecurity("bucket/object.png", &security));
}

test "s3 path security requires component containment" {
    const allowed_paths = [_][]u8{@constCast("bucket/prefix")};
    const security = ContentSecurityConfig{ .allowed_paths = &allowed_paths };

    try validateS3PathSecurity("bucket/prefix/object.png", &security);
    try std.testing.expectError(error.PathNotAllowed, validateS3PathSecurity("bucket/prefix-sibling/object.png", &security));
}

test "s3 locations always use the configured endpoint" {
    const alloc = std.testing.allocator;

    const bucket_uri = try std.Uri.parse("s3://attacker.example/bucket/object.png");
    const bucket, const key = try parseS3LocationAlloc(alloc, bucket_uri, "s3.internal.example", true);
    defer alloc.free(bucket);
    defer alloc.free(key);
    try std.testing.expectEqualStrings("attacker.example", bucket);
    try std.testing.expectEqualStrings("bucket/object.png", key);

    const endpoint_uri = try std.Uri.parse("s3://s3.internal.example/bucket/object.png");
    const endpoint_bucket, const endpoint_key = try parseS3LocationAlloc(alloc, endpoint_uri, "S3.INTERNAL.EXAMPLE", true);
    defer alloc.free(endpoint_bucket);
    defer alloc.free(endpoint_key);
    try std.testing.expectEqualStrings("bucket", endpoint_bucket);
    try std.testing.expectEqualStrings("object.png", endpoint_key);

    var resolved = try objectstore.resolveS3EndpointAlloc(alloc, "http://LOCALHOST:9000", true);
    defer resolved.deinit(alloc);
    const port_uri = try std.Uri.parse("s3://localhost:9000/bucket/object.png");
    const port_bucket, const port_key = try parseS3LocationAlloc(alloc, port_uri, resolved.endpoint, resolved.use_ssl);
    defer alloc.free(port_bucket);
    defer alloc.free(port_key);
    try std.testing.expectEqualStrings("LOCALHOST:9000", resolved.endpoint);
    try std.testing.expect(!resolved.use_ssl);
    try std.testing.expectEqualStrings("bucket", port_bucket);
    try std.testing.expectEqualStrings("object.png", port_key);
}

test "s3 credential construction validates and owns every field" {
    const alloc = std.testing.allocator;
    try std.testing.expectError(error.MissingAccessKeyId, allocS3Credentials(alloc, "s3.example.com", true, &.{}));
    try std.testing.expectError(error.MissingSecretAccessKey, allocS3Credentials(alloc, "s3.example.com", true, &.{
        .access_key_id = @constCast("key"),
    }));

    var creds = try allocS3Credentials(alloc, "s3.example.com", false, &.{
        .access_key_id = @constCast("key"),
        .secret_access_key = @constCast("secret"),
        .session_token = @constCast("token"),
    });
    defer creds.deinit(alloc);
    try std.testing.expectEqualStrings("s3.example.com", creds.endpoint);
    try std.testing.expect(!creds.use_ssl);
    try std.testing.expectEqualStrings("key", creds.access_key_id);
    try std.testing.expectEqualStrings("secret", creds.secret_access_key);
    try std.testing.expectEqualStrings("token", creds.session_token.?);
}

test "download content blocks disallowed hosts" {
    const alloc = std.testing.allocator;
    const allowed_hosts = [_][]u8{@constCast("cdn.example.com")};
    try std.testing.expectError(error.HostNotAllowed, downloadContentAlloc(alloc, "https://example.com/a.png", &.{
        .allowed_hosts = &allowed_hosts,
    }, null));
}

test "safe HTTP policy rejects an allowlisted hostname without DNS pinning" {
    const allowed_hosts = [_][]u8{@constCast("cdn.example.com")};
    const uri = try std.Uri.parse("https://cdn.example.com/image.png");
    try std.testing.expectError(error.PrivateIpBlocked, validateUrlSecurity(uri, &.{
        .allowed_hosts = &allowed_hosts,
        .block_private_ips = true,
    }));
}

test "explicit private-IP opt out permits an allowlisted hostname" {
    const allowed_hosts = [_][]u8{@constCast("cdn.example.com")};
    const uri = try std.Uri.parse("https://cdn.example.com/image.png");
    try validateUrlSecurity(uri, &.{
        .allowed_hosts = &allowed_hosts,
        .block_private_ips = false,
    });
}

test "safe HTTP policy permits an allowlisted global IP literal" {
    const allowed_hosts = [_][]u8{@constCast("8.8.8.8")};
    const uri = try std.Uri.parse("https://8.8.8.8/image.png");
    try validateUrlSecurity(uri, &.{
        .allowed_hosts = &allowed_hosts,
        .block_private_ips = true,
    });
}

test "safe HTTP policy rejects an allowlisted private IP literal" {
    const allowed_hosts = [_][]u8{@constCast("127.0.0.1")};
    const uri = try std.Uri.parse("http://127.0.0.1/image.png");
    try std.testing.expectError(error.PrivateIpBlocked, validateUrlSecurity(uri, &.{
        .allowed_hosts = &allowed_hosts,
        .block_private_ips = true,
    }));
}

test "download content blocks private ip literals" {
    const alloc = std.testing.allocator;
    try std.testing.expectError(error.PrivateIpBlocked, downloadContentAlloc(alloc, "http://127.0.0.1/test.png", &.{
        .block_private_ips = true,
    }, null));
}

test "content security defaults to blocking non-global literal hosts" {
    const security = ContentSecurityConfig{};
    const loopback = try std.Uri.parse("http://127.0.0.1/test.png");
    try std.testing.expectError(error.PrivateIpBlocked, validateUrlSecurity(loopback, &security));
    const ipv6_loopback = try std.Uri.parse("http://[::1]/test.png");
    try std.testing.expectError(error.PrivateIpBlocked, validateUrlSecurity(ipv6_loopback, &security));
    const scoped_link_local = try std.Uri.parse("http://[fe80::1%25en0]/test.png");
    try std.testing.expectError(error.PrivateIpBlocked, validateUrlSecurity(scoped_link_local, &security));

    const blocked = [_][]const u8{
        "localhost",
        "api.localhost.",
        "name.LOCAL",
        "2130706433",
        "127.1",
        "0177.0.0.1",
        "0x7f000001",
        "0x7f.0.0.1",
        "0.1.2.3",
        "10.1.2.3",
        "100.64.0.1",
        "100.127.255.254",
        "127.0.0.1",
        "169.254.1.2",
        "172.31.255.254",
        "192.0.0.1",
        "192.0.2.1",
        "192.168.1.2",
        "192.88.99.1",
        "198.18.0.1",
        "198.19.255.254",
        "198.51.100.1",
        "203.0.113.1",
        "224.0.0.1",
        "239.255.255.250",
        "240.0.0.1",
        "255.255.255.255",
        "::",
        "::1",
        "::ffff:127.0.0.1",
        "::ffff:100.64.0.1",
        "64:ff9b::7f00:1",
        "64:ff9b:1::1",
        "100::1",
        "2001:0:4136:e378:8000:63bf:3fff:fdd2",
        "2001:2::1",
        "2001:10::1",
        "2001:20::1",
        "2001:db8::1",
        "2002:7f00:1::",
        "3ffe::1",
        "3fff::1",
        "5f00::1",
        "fc00::1",
        "fe80::1",
        "fe80::1%en0",
        "fec0::1",
        "ff02::1",
    };
    for (blocked) |host| {
        try std.testing.expect(isNonGlobalHost(host));
    }

    const global = [_][]const u8{
        "8.8.8.8",
        "93.184.216.34",
        "100.128.0.1",
        "2001:4860:4860::8888",
        "2002:0808:0808::1",
        "64:ff9b::808:808",
        "::ffff:8.8.8.8",
        "123.example.com",
    };
    for (global) |host| {
        try std.testing.expect(!isNonGlobalHost(host));
    }
}
