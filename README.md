# CEG5206-Group6

Group repository for the CEG5206 team project (Group 6).

## Repository purpose

Maintain the group's project code, documentation, and reports, with regular commits throughout the project.

## Required CPU work

- [CPU implementation guide](CPU/CPU_Guide.md)
- [CPU rules](CPU_rules.md)

## Collaboration workflow

1. Clone this repository and pull the latest changes before starting work.
2. Create a separate branch for each task, such as `feature/data-preprocessing` or `docs/project-report`.
3. Make small, meaningful commits whenever a useful part of the work is complete.
4. Push changes regularly and open a pull request for teammates to review.
5. Merge reviewed changes into `main` and keep documentation up to date.

Example commands after cloning:

```bash
git pull
git switch -c feature/your-task
# Work on the project, then stage only the intended files.
git add path/to/changed-file
git commit -m "Describe the completed change"
git push -u origin feature/your-task
```

## Suggested organization

- `src/`: project source code
- `docs/`: project documentation and reports
- `tests/`: relevant tests
- `data/`: only datasets permitted for public redistribution

Create these folders as the project develops. Do not commit passwords, API keys, private student information, or course materials that are not permitted for public sharing.

## CPU Experiment Folders

| Folder | Contents |
|---|---|
| [CPU/](CPU/) | Member 1's original implementation, executables and experiment results |
| [CPU_i7_12700H_OriginalRelease/](CPU_i7_12700H_OriginalRelease/) | Local i7-12700H experiment using Member 1's original Release executable, with comparison results |
| [CPU_i7_12700H_LocalBuild/](CPU_i7_12700H_LocalBuild/) | Local i7-12700H build from Member 1's unchanged source, with experiment and comparison results |

## Project Reports

- [Three CPU experiments comparison in Chinese Word format](docs/CEG5206%E4%B8%89%E7%BB%84CPU%E5%AE%9E%E9%AA%8C%E5%8C%BA%E5%88%AB%E4%B8%8E%E5%AE%8C%E6%95%B4%E7%BB%93%E6%9E%9C%E5%AF%B9%E6%AF%94.docx)
