# PR 29 visual evidence

Screenshots for https://github.com/rubymatrix/xi-on-anything/pull/29.

- Before: code at 2506f777140c2508fca0c3fd97811f2016edbad8.
- After: code at 70dac5d4532b15bb6bbbdfdbdb80b0558d53dece.
- macOS arm64, game build 2025-11-12, local replay scene, 1920x1080 menus, 201 addons.
- Each run uses copied sandbox settings. Images are real client frame captures, cropped and labeled; no game UI content was retouched.
- display.png, menus.png, modern.png, addons.png: before at left, after at right; captured after opening the page without moving the page cursor.
- focus-press.png: before in the top row, after in the bottom row; second-row focus, first-row ON focus, first-row ON pressed.
- wheel.png: before in the top row, after in the bottom row; initial list and five wheel-down events (event 23) delivered directly to the menu handler. This demonstrates handler behavior, not operating-system mouse delivery.
- scrollbar-corner.png: original user-provided screenshot of an intermediate native-scrollbar build on the left, current build on the right. Both show rows 24-33; crops enlarged 4x without smoothing. This intermediate artifact was absent from the PR base and corrected during development.

Only screenshots and this provenance note are stored on this evidence branch. No changes to the PR code branch.
