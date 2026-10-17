"""Operation of equipment and accounts in the browser (step P14.8, REQ-WEB-9, ADR-0019).

Requirements: the operation panel lists the devices, their state and the actions of the elements; when the
operation is enabled, devices are discovered and associated, and every action on equipment is confirmed in a
dialog naming the device; the emergency stop is always at hand; without the operation, actions only change the
simulation and need no confirmation; logging in shows the role of the user, logging out ends the session.

The equipment is the LumosLabs LL-LS100 emulator (ADR-0014), reached over TCP.
"""

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def open_bench(page, mode):
    summary = page.request.post("/api/contexts", data={"name": "bench", "mode": mode}).json()
    params = {"velocity": "50 mm/s", "acceleration": "500 mm/s2"}
    element = {"type_id": "lumos.lab.LinearStage", "name": "stage", "params": params}
    assert page.request.post(f"/api/contexts/{summary['id']}/elements", data=element).ok
    page.evaluate(f"localStorage.setItem('lumos.context', '{summary['id']}')")
    page.reload()
    expect(page.locator('.react-flow__node[data-id="stage"]')).to_be_visible()


def test_equipment_is_discovered_associated_and_driven_with_a_confirmation(
    browser, operation_server, stage_emulator
):
    context = browser.new_context(
        base_url=operation_server, locale="en-GB", viewport={"width": 1440, "height": 900}
    )
    page = context.new_page()
    page.goto("/")
    open_bench(page, "operation")
    page.get_by_test_id("tab-operation").click()
    panel = page.get_by_test_id("operation")
    panel.get_by_test_id("device-urls").fill(stage_emulator.url)
    panel.get_by_test_id("discover").click()
    expect(panel.get_by_test_id("discovered")).to_contain_text("LL-LS100")
    panel.get_by_test_id("associate").click()
    expect(panel.get_by_test_id("devices")).to_contain_text("stage")
    panel.get_by_test_id("action-stage-home").click()
    dialog = page.get_by_test_id("confirm-action")
    expect(dialog).to_contain_text("LL-LS100")
    dialog.get_by_test_id("dialog-confirm").click()
    expect(page.get_by_test_id("toast-success")).to_contain_text("home")
    page.get_by_test_id("tab-journal").click()
    expect(page.get_by_test_id("journal")).to_contain_text("stage.home")
    page.get_by_test_id("emergency-stop").click()
    expect(page.get_by_test_id("toast-info")).to_be_visible()
    context.close()


def test_without_the_operation_actions_change_the_simulation_only(page):
    page.goto("/")
    open_bench(page, "simulation")
    page.get_by_test_id("tab-operation").click()
    panel = page.get_by_test_id("operation")
    expect(panel).to_contain_text("disabled")
    expect(panel.get_by_test_id("discover")).to_have_count(0)
    panel.get_by_test_id("action-stage-home").click()
    expect(page.get_by_test_id("confirm-action")).to_have_count(0)  # nothing real is driven
    expect(page.get_by_test_id("toast-success")).to_contain_text("simulation")


def test_logging_in_shows_the_role_and_logging_out_ends_the_session(browser, users_server):
    context = browser.new_context(
        base_url=users_server, locale="en-GB", viewport={"width": 1440, "height": 900}
    )
    page = context.new_page()
    page.goto("/")
    page.get_by_test_id("login").click()
    dialog = page.get_by_test_id("login-dialog")
    dialog.get_by_test_id("login-user").fill("simon")
    dialog.get_by_test_id("login-password").fill("wrong")
    dialog.get_by_test_id("login-submit").click()
    expect(dialog).to_contain_text("wrong user name or password")
    dialog.get_by_test_id("login-password").fill("simulator password")
    dialog.get_by_test_id("login-submit").click()
    expect(page.get_by_test_id("user")).to_contain_text("simon")
    expect(page.get_by_test_id("user")).to_contain_text("simulator")
    page.get_by_test_id("user").click()
    page.get_by_test_id("logout").click()
    expect(page.get_by_test_id("login")).to_be_visible()
    context.close()


def test_the_ssh_login_shows_the_command_to_sign_the_challenge(browser, users_server):
    context = browser.new_context(
        base_url=users_server, locale="en-GB", viewport={"width": 1440, "height": 900}
    )
    page = context.new_page()
    page.goto("/")
    page.get_by_test_id("login").click()
    dialog = page.get_by_test_id("login-dialog")
    dialog.get_by_test_id("login-tab-ssh").click()
    dialog.get_by_test_id("login-user").fill("simon")
    dialog.get_by_test_id("login-challenge").click()
    expect(dialog.get_by_test_id("ssh-command")).to_contain_text("ssh-keygen -Y sign -n lumos")
    expect(dialog.get_by_test_id("ssh-command")).to_contain_text("lumos-login:")
    context.close()
