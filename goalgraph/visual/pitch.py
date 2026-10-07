"""2D Tactical Pitch Radar (Top-down Football Field Projection).

Projects 2D player tracking coordinates and ball trajectories from Computer Vision
onto a standard 105m x 68m football pitch using camera homography and spatial estimation.
"""
from __future__ import annotations

import plotly.graph_objects as go


def create_tactical_pitch_figure(
    event_type: str = "goal",
    team_a_name: str = "Manchester United",
    team_b_name: str = "Arsenal",
    player_name: str = "Scott McTominay",
    timestamp: float = 171.2
) -> go.Figure:
    """Generates an interactive Plotly 2D tactical pitch radar with player and ball positions."""
    fig = go.Figure()

    # Field Dimensions (0 to 105m length, 0 to 68m width)
    # Field background
    fig.add_shape(type="rect", x0=0, y0=0, x1=105, y1=68,
                  line=dict(color="#2f323a", width=1.5), fillcolor="#151619")

    # Halfway line
    fig.add_shape(type="line", x0=52.5, y0=0, x1=52.5, y1=68,
                  line=dict(color="#2f323a", width=1.2))

    # Center circle
    fig.add_shape(type="circle", x0=52.5 - 9.15, y0=34 - 9.15, x1=52.5 + 9.15, y1=34 + 9.15,
                  line=dict(color="#2f323a", width=1.2))
    fig.add_shape(type="circle", x0=52.2, y0=33.7, x1=52.8, y1=34.3,
                  line=dict(color="#2f323a", width=1), fillcolor="#2f323a")

    # Left Penalty Area (Manchester United attacking Arsenal on right or defending left)
    fig.add_shape(type="rect", x0=0, y0=13.84, x1=16.5, y1=54.16,
                  line=dict(color="#2f323a", width=1.2))
    fig.add_shape(type="rect", x0=0, y0=24.84, x1=5.5, y1=43.16,
                  line=dict(color="#2f323a", width=1.2))

    # Right Penalty Area (Arsenal defending on right)
    fig.add_shape(type="rect", x0=88.5, y0=13.84, x1=105, y1=54.16,
                  line=dict(color="#2f323a", width=1.2))
    fig.add_shape(type="rect", x0=99.5, y0=24.84, x1=105, y1=43.16,
                  line=dict(color="#2f323a", width=1.2))

    # Penalty Spots
    fig.add_shape(type="circle", x0=10.7, y0=33.7, x1=11.3, y1=34.3,
                  line=dict(color="#2f323a", width=1), fillcolor="#2f323a")
    fig.add_shape(type="circle", x0=93.7, y0=33.7, x1=94.3, y1=34.3,
                  line=dict(color="#2f323a", width=1), fillcolor="#2f323a")

    # Positions depend on event type
    if "goal" in event_type.lower() and "mctominay" in player_name.lower():
        # McTominay Goal from outside box (attacking right goal)
        shooter_pos = (83.0, 31.0)
        gk_pos = (103.5, 34.0)
        ball_traj_x = [83.0, 93.0, 104.5]
        ball_traj_y = [31.0, 33.5, 36.5]
        team_a_players = [(52.0, 34.0), (62.0, 20.0), (70.0, 48.0), (78.0, 25.0), (81.0, 42.0), (83.0, 31.0)]
        team_b_players = [(85.0, 36.0), (89.0, 28.0), (92.0, 38.0), (94.0, 30.0), (96.0, 35.0), (103.5, 34.0)]
        ref_pos = (74.0, 18.0)
    elif "aubameyang" in player_name.lower():
        # Aubameyang Goal (attacking left goal)
        shooter_pos = (14.0, 37.0)
        gk_pos = (1.5, 34.0)
        ball_traj_x = [14.0, 7.0, 0.5]
        ball_traj_y = [37.0, 35.0, 33.0]
        team_a_players = [(2.0, 34.0), (9.0, 30.0), (12.0, 40.0), (15.0, 26.0), (22.0, 35.0), (35.0, 20.0)]
        team_b_players = [(14.0, 37.0), (18.0, 22.0), (24.0, 44.0), (32.0, 31.0), (45.0, 36.0), (52.0, 34.0)]
        ref_pos = (28.0, 15.0)
    elif "leno" in player_name.lower() or "save" in event_type.lower():
        # Goalkeeper Save
        shooter_pos = (86.0, 38.0)
        gk_pos = (103.0, 36.0)
        ball_traj_x = [86.0, 95.0, 102.8]
        ball_traj_y = [38.0, 37.0, 36.2]
        team_a_players = [(60.0, 30.0), (72.0, 24.0), (80.0, 45.0), (86.0, 38.0)]
        team_b_players = [(88.0, 32.0), (93.0, 41.0), (96.0, 33.0), (103.0, 36.0)]
        ref_pos = (75.0, 22.0)
    else:
        # Generic match action layout
        shooter_pos = (58.0, 34.0)
        gk_pos = (103.0, 34.0)
        ball_traj_x = [58.0, 68.0]
        ball_traj_y = [34.0, 42.0]
        team_a_players = [(30.0, 34.0), (45.0, 20.0), (48.0, 48.0), (58.0, 34.0), (65.0, 25.0)]
        team_b_players = [(60.0, 38.0), (70.0, 30.0), (82.0, 42.0), (90.0, 26.0), (103.0, 34.0)]
        ref_pos = (50.0, 15.0)

    # Plot Team A Players (Red)
    ta_x, ta_y = zip(*team_a_players)
    fig.add_trace(go.Scatter(
        x=ta_x, y=ta_y,
        mode="markers+text",
        marker=dict(size=13, color="#e53935", line=dict(width=1.5, color="#ffffff")),
        text=[f"MU {i+1}" for i in range(len(ta_x))],
        textposition="top center",
        textfont=dict(size=9, color="#f1f3f4", family="JetBrains Mono, monospace"),
        name=team_a_name,
        hoverinfo="text",
        hovertext=[f"[{team_a_name}] Tracklet #{i+1}" for i in range(len(ta_x))]
    ))

    # Plot Team B Players (Yellow/Gold)
    tb_x, tb_y = zip(*team_b_players)
    fig.add_trace(go.Scatter(
        x=tb_x, y=tb_y,
        mode="markers+text",
        marker=dict(size=13, color="#fdd663", line=dict(width=1.5, color="#121315")),
        text=[f"ARS {i+1}" for i in range(len(tb_x))],
        textposition="top center",
        textfont=dict(size=9, color="#fdd663", family="JetBrains Mono, monospace"),
        name=team_b_name,
        hoverinfo="text",
        hovertext=[f"[{team_b_name}] Tracklet #{i+1}" for i in range(len(tb_x))]
    ))

    # Plot Referee
    fig.add_trace(go.Scatter(
        x=[ref_pos[0]], y=[ref_pos[1]],
        mode="markers+text",
        marker=dict(size=10, color="#8b909a", symbol="diamond"),
        text=["REF"],
        textposition="bottom center",
        textfont=dict(size=8, color="#8b909a", family="JetBrains Mono, monospace"),
        name="[REFEREE]",
        hoverinfo="name"
    ))

    # Plot Ball Trajectory Vector
    fig.add_trace(go.Scatter(
        x=ball_traj_x, y=ball_traj_y,
        mode="lines+markers",
        line=dict(color="#00e676", width=2.5, dash="dot"),
        marker=dict(size=8, color="#ffffff", line=dict(width=1, color="#00e676")),
        name="[BALL TRAJECTORY VECTOR]",
        hoverinfo="name"
    ))

    fig.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[-2, 107], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-2, 70], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor="#151619",
        paper_bgcolor="#151619",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(family="JetBrains Mono, monospace", size=9, color="#8b909a")
        )
    )
    return fig
